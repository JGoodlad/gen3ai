# The AlphaStar-style path: what a league means at our compute, and the order to build it

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** When a decision, build or read differs from
this doc, update the doc in the same commit and say in the Decision record (§8) what changed and why.

**Status (2026-10-20): DESIGN; LG-0 READ (X55, [`x55_league_headroom_2026-10-20/`](../research_state/measurements/x55_league_headroom_2026-10-20/README.md), the
same day: §2.4).** Nothing else here is built, registered or launched. The roadmap in §6 is sequenced AFTER
X35 batch 1 ([`design_compute_efficiency.md`](design_compute_efficiency.md) §5.6, launch sheet
[`BATCH1_LAUNCH_SHEET.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md)), whose
toy block is on the GPU now. Every launch needs the orchestrator's GO, and a design block is not a launch command.

**Trigger.** This is goal 4 of the owner's 2026-10-17 order: (1) the strongest baseline, (2) vs Metamon, (3) where we
are, (4) **classic AlphaStar-style approaches (league, PFSP, exploiters) as the plateau-breakers**, (5) what search
needs, (6) collapse and counterfactuals. The orchestrator asked what "AlphaStar-style" concretely means for OUR system
at OUR compute, and in what order to build it.

**Our compute.** One RTX 3080 Ti-class GPU, shared with everything else. Production trains at ≈ 1,334 steps/s of wall
time, eval included (X26 150M → 180M, MEASURED, `design_compute_efficiency.md` §5.6.1). The toy (T-ENT) trains at
≈ 4,500–5,000 steps/s (≈ 5.8 GPU-h per 100M). One 20M production fork costs ≈ 4.4 GPU-h, and one 5M exploiter
≈ 1.1 GPU-h. For scale: AlphaStar trained each of its 12 learning agents on 32 TPUv3 devices for 44 days (§1.1), roughly
three to four orders of magnitude more compute than one X26-length run here (ESTIMATED).

---

## 0. One screen

- **What "AlphaStar-style" means here.** AlphaStar's league has four mechanisms (§1.1):
  1. a growing ARCHIVE of frozen past players that is never pruned;
  2. PFSP, which samples opponents by how hard they are, read off a payoff matrix kept for every pair;
  3. EXPLOITERS that best-respond to the main agent (main exploiters) or to the whole league (league exploiters),
     are reset, and whose snapshots join the league;
  4. the main agent's fixed opponent MIX: self-play, PFSP over the past, and a forgotten-or-exploiter slice.

  We have all four in a reduced form, as flags (§3). X35's A1 is the archive, A2 is PFSP over the RECENT pool, A3 is
  main exploiters as one frozen round, and A6 is league exploiters. Two pieces are missing: AlphaStar's PFSP is over
  the ARCHIVE, and the league GROWS in one run. We have neither, and X35 has no arm for either.
- **What we measured about our game.** X26 climbs transitively (P\* beats its 25M self by +18.8 pp in-house). At the
  plateau it sits on a small but REAL cyclic component: about 3 pp of score per edge in the last 80M and about 4 pp
  over the whole archive, MATERIAL against the 1.7 pp bar at 28 of 37 checks. The posterior Nash support is 6–9 nodes
  from 140M on. The newest node scores ≈ 0.53 on average against nodes 20–190M older (§2).
- **What that predicts.** The plateau looks like the WIDE TOP of a spinning top: a cluster of ≈ 8 nearly equal
  strategies that beat each other a little. From that cluster:
  - Archive and PFSP levers can buy about the CYCLIC SIZE, i.e. **≈ 0–3 pp** on X35's population score.
  - X35's R1 reads GAIN only above **≈ 3.4–5.4 pp** (§2.3, FINDING F-AS-1).
  - **The modal outcome of A1 and A2 is therefore UNDECIDED, not GAIN.** The roadmap is built so that an UNDECIDED
    still moves it.
  - **LG-0 MEASURED it (§2.4):** the archive's Nash sits in C0's own recent pool (0.90 of its posterior mass), A1's
    nodes' neighbourhoods hold 0.19, P\* scores 0.45–0.61 against every archive node it was measured on (18 cells,
    none below 0.5 at 95 %) and 0.514 ± 0.014 against the Nash mixture, and PFSP over A1's archive is near-uniform
    (max / min weight 1.3). The archive holds no hole for a league to repair; registered predictions:
    [`PREDICTIONS.md`](../research_state/measurements/x55_league_headroom_2026-10-20/PREDICTIONS.md).
  - Exploitability (the best-response gap) is the meter on which population methods have MOVED in this project. The
    old lineage's exploiter loop cut the gap by −10.0 and −8.25 pp in two rounds (§2.2).
- **The literature at small compute (§1).** Most published leagues are the price of AlphaStar's scale, and the
  cost-cutting successors cut the NUMBER and COST of exploiters, not the idea. The league-free route is regularised
  self-play (R-NaD / MMD; DeepNash reached expert level in Stratego with one agent and no league), which is X35's A4.
  Neural population learning (NeuPL) is how a league fits in one network; it is a later option, not a first step.
- **The roadmap (§6), ranked, after X35 batch 1:**

  | # | step | cost |
  |---|---|---|
  | 1 | **LG-0** the league HEADROOM read, on CPU, now: the Nash cluster, the archive's exploitability of P\*, predictions registered before A1 reads | 0 GPU-h, ≈ 0.5 agent-day |
  | 2 | **LG-1** MAIN EXPLOITERS (X35-A3), with exploitability as co-primary; its R0 is now X35's registered R0-prod (AM-R0, run right after P1's C0 seeds) | ≈ 18 GPU-h, ≈ 0.25 agent-day |
  | 3 | **LG-2** PFSP OVER THE ARCHIVE (A1 + `--stable-opponent-pfsp`), AlphaStar's main-agent slice, no build | ≈ 14.4 GPU-h, ≈ 0.25 agent-day |
  | 4 | **LG-3** the LEAGUE-LITE continuation, ON THE TOY FIRST: a growing archive with a live-opponent cap, PFSP over it, and a reset exploiter round every 50M, run long against a matched control | builds ≈ 3–4 agent-days; ≈ 35 GPU-h (toy), ≈ 105 (production) |
  | 5 | **LG-4** the TRAILING MAGNET (X35-A4, R-NaD / MMD), the league-free alternative; it rises to #3 if the population levers read NULL | ≈ 15 GPU-h, ≈ 2 agent-days (T56) |
  | 6 | **LG-5** the MULTI-LINEAGE league (production, toy and the closing-test seeds as opponents of each other; X29's basin probe), CONDITIONAL | ≈ 15–30 GPU-h, ≈ 1 agent-day |

  NeuPL-style conditional populations, PSRO with a Nash meta-solver over many best responses, and XDO are rejected
  for now (§5).
- **Decision branches (§7).** A1's result decides whether the archive gets more machinery: GAIN means LG-2 and LG-3,
  NULL or LOSS means skip LG-2 and move LG-4 up. A3's R2 decides whether exploiters are worth automating (LG-3's
  exploiter rounds). D1 decides whether the league's diversity should come from TEAMS rather than policies. X47
  decides whether capacity outranks the league. G1 decides how many seeds every later arm needs.

---

## 1. The literature, applied to us

**Scope and sourcing.** Two literature workers checked the claims below on 2026-10-20, from primary sources where they
could. The AlphaStar paper was read in its accepted-manuscript PDF (DeepMind's media site), because Nature's page is
paywalled. The arXiv papers were read on arXiv, the full text where needed. **VERIFIED** means the claim was read in
the primary source. **UNVERIFIED** says what is missing. `design_evaluation.md` §0a already covers the
rating-and-evaluation literature (Nash averaging, HodgeRank, spinning tops, α-Rank, the Dota 15–85 % band); this
section covers how a league TRAINS.

### 1.1 AlphaStar's league (Vinyals et al., *Nature* 575:350–354, 2019, doi:10.1038/s41586-019-1724-z)

| claim | status | for us |
|---|---|---|
| Per race: 1 main agent, 1 main exploiter, 2 league exploiters. That is 12 learning agents in all | VERIFIED (Methods, "Infrastructure") | we have one main agent. The team archetype is the closest analogue to a race (§3) |
| Main agent: 35 % self-play, 50 % PFSP against ALL past players, 15 % PFSP against forgotten main players it can no longer beat and past main exploiters (falling back to self-play when there are none). Main agents never reset | VERIFIED (Methods, "Populating the League") | A1 ≈ the 50 % slice without PFSP. LG-2 adds the PFSP |
| PFSP: P(B) ∝ f(P[A beats B]). f_hard(x) = (1 − x)^p is the default, and f_var(x) = x(1 − x) is for main exploiters and struggling main agents | VERIFIED. **The value of p is UNVERIFIED**: the paper says only that p ∈ ℝ⁺ "controls how entropic" the distribution is. DI-engine's reimplementation uses a squared form, suggesting p = 2 (secondary) | our stable PFSP is f_hard with p = 1, floored at 0.05 |
| Main exploiter: plays the CURRENT main agents. Half the time, when its win rate is below 20 %, it plays main-agent-derived players under f_var. It is snapshotted at > 70 % against all 3 main agents, or at a timeout, and then ALWAYS reset to the supervised parameters | VERIFIED. The timeout exponent is garbled in the PDF; 4·10⁹ steps is secondary (TStarBot-X §4.5, DI-engine) | ours is one round against a FROZEN P\*, never reset (F24) |
| League exploiter: PFSP against the whole league; snapshotted at > 70 % or at a timeout (2·10⁹, secondary); reset with probability 25 % | VERIFIED (the rule); the timeout is secondary | A6 is one frozen round of this |
| Main-agent snapshots every 2·10⁹ steps (the exponent is garbled; DI-engine is the secondary). About 900 distinct players by the end | VERIFIED (900); the cadence is secondary | our archive would add one node per 10M (§4.4) |
| A central coordinator keeps a payoff-matrix estimate, and CPU evaluators supplement it | VERIFIED. **The estimator (EMA or counts) is UNVERIFIED** | ours: the eval ledger (counts) plus PFSP's EMA |
| Compute: each agent on 32 TPUv3 for 44 days, 16,000 concurrent games, ≈ 50,000 agent steps/s per learner. Supervised init and a KL to the supervised policy, conditioned on a human statistic z | VERIFIED | ≈ 1,334 steps/s for us. No human prior (X50) |
| **League ablation** (Fig. 3, Protoss vs Protoss, one map, 10¹⁰ steps): main agents alone reach test Elo 1540 (relative population performance 6 %), + main exploiters 1693 (35 %), + league exploiters 1824 (62 %) | VERIFIED | **exploiters were the largest single component**. That is why LG-1 is the first GPU step |
| **Opponent-mix ablation** (Fig. 3): test Elo PFSP + SP 1540, SP 1519, PFSP 1273, FSP 1143. Minimum win rate against past versions 71 / 46 / 70 / 69 % | VERIFIED | pure self-play is strong but FORGETS (46 %). Mixing in PFSP fixes the forgetting at no Elo cost. Pure PFSP is weak. A1's 50/50 shape matches the best row |

### 1.2 The successors that cut the cost

| paper | what it changed, and the result | status | for us |
|---|---|---|---|
| **TStarBot-X** (Han et al., arXiv:2011.13729, 2020/21) | A **faithful AlphaStar scale-down** (1 main, 2 main exploiters, 2 league exploiters, at 1/30 of AlphaStar's consumption speed) **COLLAPSED to rush strategies within days**. AlphaStar's exploiter period cost over 10 days, and shorter periods gave exploiters too weak to catch the main agent. The fix, "Diversified League Training", used specific exploiters started from strategy-specific fine-tuned models, evolutionary exploiters, and resets to a past player the main agent wins 20–50 % against. It also added **DAPO, a KL to the agent's own 12-hour-old snapshot**. The main agent's mix was 25 / 60 / 15. Compute: 144 V100s and 13,440 CPU cores for 57 days | VERIFIED (§1, §4.5, Table 1, §5.1.3) | **the strongest warning for our scale**: copying AlphaStar's roles at a fraction of its compute fails. Exploiters must be CHEAP and DIVERSE, and must start from something that can already compete. DAPO is a trailing magnet (A4 / LG-4) |
| **SCC** (Wang et al., ICML 2021, arXiv:2012.13169) | An "order of magnitude less computation": ≈ 1,000 games and ≈ 800 agent steps/s per agent. **Agent branching**: a new exploiter starts from the CURRENT main agent, with a dense reward toward a chosen strategy statistic, so it specialises faster | VERIFIED (abstract, §6.2, App. B). League size and GPU count UNVERIFIED | our exploiters already start from P\* (a fork). That is SCC's branching, and it is the cheap form |
| **ROA-Star** (Huang et al., NeurIPS 2023; arXiv / OpenReview id **UNVERIFIED**) | In a replicated AlphaStar, exploiters LOSE their ability to find weaknesses later in training. Fixes: goal-conditioned exploiters (exploitative ones reset to the current main agent, explorative ones to the supervised model) and an opponent-modelling auxiliary. 10-day main agents: worst-case win rate 44 % against 1 % for the replica, average 84.1 % against 70.2 %. Compute: 64 V100s per agent | VERIFIED (abstract, §3, Tables 1 and 3, Fig. 6) | exploiter decay is why LG-3 RESETS exploiters every round. Our nearest "goal" is the team: team-pinned exploiters (X42) |
| **Minimax Exploiter** (Bairamian et al., AAMAS 2024, arXiv:2311.17190) | The exploiter's reward is shifted by the frozen main agent's value: R − αγ(1 − d)·max_a Q_main(s′, a). Measured in wall clock, on one machine: 16 converged exploiters against 13 for vanilla in 100 h (For Honor) | VERIFIED (§4.2 Eq. 3, §5) | our critic is a win probability, so the shaping is available. It is a COST-REDUCER for LG-3's repeated rounds (≈ 20 % more exploiters per hour in their setting), not a first step |
| **AlphaStar Unplugged** (Mathieu et al., arXiv:2308.03526) | Offline RL on human replays. The best offline agents win ≥ 90 % against the AlphaStar behaviour-cloning agent | VERIFIED (abstract, conclusion). Compute UNVERIFIED | the Metamon route (offline from replays), not ours. It belongs to X50 |
| **DI-star** (OpenDILab) | An open AlphaStar reimplementation. RL used 128 A100s for about 2 months; only SUPERVISED training is documented on one 8 GB card | VERIFIED (README, docs). No small-compute league claim | no small-compute league to copy |
| **OpenAI Five** (Berner et al., arXiv:1912.06680, App. N) | 80 % of games against the latest self, 20 % against the past. Past opponents are sampled by softmax over quality scores; a win by the current agent lowers that opponent's score, q_i ← q_i − η/(N·p_i), η = 0.01 | VERIFIED | a cheaper, online PFSP form that needs no full payoff matrix. It is an alternative weighting for LG-2 if f_hard's win-rate estimates are too noisy |

### 1.3 Population-based game theory: PSRO and its descendants

| paper | idea | status | for us |
|---|---|---|---|
| **PSRO** (Lanctot et al., NeurIPS 2017, arXiv:1711.00832) | Grow a population: a meta-solver gives a mixture over the current policies, an oracle trains a best response to it, the best response is added, and the meta-game is re-solved. Double oracle is the Nash meta-solver; fictitious play is the uniform one. DCH is the parallel approximation | VERIFIED (§3, Alg. 1) | our population loop IS one PSRO iteration with a uniform-ish meta-solver. The difference is that PSRO's answer is the MIXTURE, while we must deploy ONE network, which is why we fold the population back into the main agent (AlphaStar's move) |
| **Pipeline PSRO** (McAleer et al., NeurIPS 2020, arXiv:2006.08555) | Parallel best responses at several levels; only the lowest active one is fixed once it plateaus. 71 % average against existing Barrage Stratego bots | VERIFIED. Its Barrage compute is UNVERIFIED | parallelism buys nothing on ONE GPU |
| **XDO / NXDO** (McAleer et al., NeurIPS 2021, arXiv:2103.06426) | Mixes best responses at every INFOSTATE. Iterations are bounded linearly in infostates, against exponentially for double oracle | VERIFIED (Props. 2–3, §5) | it needs an infostate-indexed restricted game. Ours is far too large, and our learner does not expose one |
| **Anytime PSRO / Self-Play PSRO** (arXiv:2201.07700, arXiv:2207.06541) | Make PSRO's exploitability fall monotonically, and add an off-policy stochastic policy each iteration to converge in fewer iterations | VERIFIED (one line each) | refinements of a loop we do not run at scale |
| **NeuPL** (Liu, Marris, Hennes, Merel, Heess, Graepel; ICLR 2022, arXiv:2202.07415) | ONE network Π(·\|s, σ_i) represents the whole population, each policy conditioned on its meta-game row. Skills transfer between best responses, so novel strategies get EASIER to find as the population grows. It uses "the computational infrastructure of simple self-play" (no numeric compute figure) | VERIFIED (§1, §2.2, Fig. 6). The brief's author list was wrong and is corrected here | **the right end state if a league outgrows the fan-out budget**: one network, one forward, any number of "members". It is a new architecture input and a fresh lineage, so it is DEFERRED (§5) |
| **Simplex-NeuPL** (Liu, Lanctot, Marris, Heess; ICML 2022, arXiv:2205.15879) | A conditional network trained on Dirichlet-sampled mixtures, so it best-responds to ANY mixture. The Bayes-optimality is shown empirically (goofspiel, running-with-scissors), with no theorem | VERIFIED | same deferral |
| **PSRO_rN, effective diversity** (Balduzzi et al., ICML 2019, arXiv:1901.08106) | Rectified Nash-response PSRO beats Nash PSRO, and both far beat self-play on Blotto and differentiable Lotto. PSRO_U (best response to uniform) is about as good as Nash | VERIFIED (§3, §5) | in a toy cyclic game the meta-solver matters little and population matters a lot. That supports "uniform archive first" (A1) and weighting second (LG-2) |
| **Spinning tops** (Czarnecki et al., arXiv:2004.09468; NeurIPS 2020 per the review in `design_evaluation.md` §0a) | Fixed-memory fictitious play converges in a layered game of skill if the population is at least the size of the lowest occupied layer (Prop. 3). A population that contains a full Nash cluster gives transitive improvement when beaten (Thm 3). Empirically there is a phase change at a critical population size: below it training cycles, above it it converges | VERIFIED (§4, §6) | **our cluster is ≈ 8 nodes (§2.1)**. The recent pool of 14–20 is large enough IF it holds the cluster's strategies. LG-0 checks that |
| **Policy-space diversity** (Yao et al., arXiv:2306.16884; NeurIPS 2023 per the citation, venue UNVERIFIED from the PDF) | Common diversity metrics do not guarantee a better Nash approximation. PSD-PSRO gives less exploitable populations | VERIFIED (claim) | if LG-3 adds diversity bonuses later, measure exploitability, not diversity |

### 1.4 Single-agent equilibrium training, the league-free route

| paper | idea | status | for us |
|---|---|---|---|
| **DeepNash / R-NaD** (Perolat et al., *Science* 378, 2022, doi:10.1126/science.add4679, arXiv:2206.15378) | ONE agent, self-play, no search and **no league**. Regularised Nash dynamics: a regularisation policy, a dynamics step and a fixed-point update. It wins > 97 % against existing Stratego bots and ranked 3rd on Gravon (42 / 50 games, rating 1799) | VERIFIED. Compute UNVERIFIED | the existence proof that an imperfect-information game reaches expert play WITHOUT a population. That is A4 / LG-4 |
| **MMD** (Sokota et al., ICLR 2023, arXiv:2206.05825) | Magnetic mirror descent: linear last-iterate convergence to quantal-response equilibria. **As PPO with a reverse-KL magnet** it reaches lower approximate exploitability than PPO and NFSP on Abrupt Dark Hex and Phantom Tic-Tac-Toe, while plain PPO "does not reliably reduce approximate exploitability over time" | VERIFIED (§5) | the closest published form to our PPO. It is what T56 implements |

### 1.5 What has been measured at small compute, and in Pokémon

- **VGC-Bench** (Angliss, Cui, Hu, Rahman, Stone; AAMAS 2026, arXiv:2506.10326) is the only published comparison of
  self-play, fictitious play (uniform past checkpoints) and double oracle (an LP Nash over the pool) in a Pokémon game.
  It used PPO at ≈ 5M timesteps and 5 seeds. VERIFIED (§4.2.1, Tables 2 and 8–11):
  - With ONE team, self-play beats the population methods: SP vs DO 0.707, SP vs FP 0.587, BCSP vs BCDO 0.744.
  - At 4, 16 and 64 teams every SP / FP / DO cell is 0.47–0.57, within about one ±0.03 error bar of 0.5. **No
    population method beats self-play.**
  - "In almost all cases, all agents are approximately 100 % exploitable."
  - Behaviour-cloning init helped more than any opponent-selection rule.

  Our runs are ≈ 50× longer and our pool far wider. So this is a WEAK prior that population methods matter little in
  Pokémon at small budgets, and a strong reminder that exploitability, not the head-to-head, is where they would
  show.
- **Opponent sampling from the history** (Bansal et al., ICLR 2018, arXiv:1710.03748, §5.4 Table 1; continuous-control
  games): sampling only the latest opponent was WORST. Uniform over the whole history (Ant: E[win] 0.50 against 0.34)
  or over its second half (Humanoid: 0.63 against 0.25) was best. VERIFIED, but the table's column mapping is
  garbled in the extracted text, so it is not fully certain. This supports an ARCHIVE over a pure recency window.
- **A generalised self-play framework** (Hernandez et al., CoG 2019, doi:10.1109/CIG.2019.8848006; **the CoG text is
  UNVERIFIED**; the companion arXiv:2006.04471 was read). In repeated RPS with PPO, uniform sampling over the FULL
  history improves monotonically, while windowed sampling cycles. PSRO's oracle hyperparameters were fragile. It is
  qualitative, with no numbers.
- **Metamon** (Grigsby et al., RLC 2025, arXiv:2504.04395) fine-tuned on its own checkpoints' games, with no league.
  The gain was "significantly better against itself" but "inconsistent" against real players. **No numeric effect is
  stated (UNVERIFIED as a number;** read through an HTML summary).
- **The published Showdown bots** (`../research_state/literature/showdown_bots_landscape_2026-10-18.md`) reached #1
  with PPO self-play against a pool of past selves. Only one (antonius1, Reg M-B, peak 1365, not top) reports
  FSP / double-oracle leagues, with a behaviour-cloned human model inside. None reports an AlphaStar-style league
  with exploiters.

**What the literature says for our compute, in four lines.**
1. Exploiters were AlphaStar's largest component, but at a fraction of its compute a faithful copy collapses. Cheap
   exploiters branched from the current agent and reset often (SCC, ROA-Star) are the form that survives.
2. An archive beats a recency window against forgetting (AlphaStar's SP-vs-PFSP minimum win rate; Bansal et al.;
   Hernandez et al.). Hardness weighting helps only when the archive is diverse.
3. A league is not the only route to an equilibrium: R-NaD / MMD get there with one agent. TStarBot-X's DAPO is the
   same idea used as a league stabiliser.
4. In Pokémon at small budgets, nobody has shown a population method beating self-play on strength (VGC-Bench).
   Exploitability is where the effect, if any, should first appear.

---

## 2. What we know about our own game's population structure

### 2.1 Cycling at the plateau (MEASURED, descriptive)

The source is the cycle monitor's retro read on X26's in-run snapshot ladder
([`../research_state/measurements/cycle_monitor_retro_2026-10-19/`](../research_state/measurements/cycle_monitor_retro_2026-10-19/README.md);
`design_evaluation.md` §8.2 "The size bar"; ledger 2026-10-19 *THE CYCLE MONITOR*, *THE SIZE BAR*). It covers
72 snapshots from 2M to 376M and 1,150 mirrored lines. Each pool line has 50 pairs (a tenth of the registered 500),
and the edges are the ones the pool chose.
- **Accumulated over the whole archive:** the Hodge residual sits at the bootstrap floor (p = 0.0005) at every check.
  The excess cyclic RMS is 3.9–4.8 pp of score, and its cyclic share is 0.23–0.25.
- **Inside the last 80M:** the excess cyclic RMS is 2.7–4.2 pp from 170M on. Under the size bar it reads MATERIAL at
  28 of 37 checks (270M: 3.3 pp [1.9, 4.6]) and too wide to call at the other 9. It is never NEGLIGIBLE.
- **The Nash cluster.** The point Nash support is 6–7 nodes and the posterior support 7.7–8.5 at the 280M and 370M
  checks. The newest node is in the support with P ≈ 0.82–0.86.
- **The flat top, made of near-equal opponents.** At the 280M check the newest node (268M) scored 0.46–0.62 against
  the 19 nodes 14–134M older, mean ≈ 0.54 (each at 50–100 pairs, SE ≈ 3–5 pp). At the 370M check the newest node
  (352M) scored 0.41–0.62, mean ≈ 0.53. Its lag curve FLAGS: it scores 0.62 against the 200M node and 0.45 against
  the 194M node.
- **Progress is still large on the long scale.** P\* (280M) beats X26 @ 25M by +18.8 pp on home72 and +14.8 pp on
  unseen teams (X37b, ledger 2026-10-19).
- **The toy (X34)** has registered monitor rows only to 60M: NONE at 30M and 40M, SUSPECTED at 50M and 60M with sizes
  ≈ 2.5 pp, INCONCLUSIVE. Its plateau region is not read yet.

**Reading.** This is the spinning-top picture (Czarnecki et al. 2020, §1.3): transitive progress on the long axis,
and at the top a cluster of about 8 strategies of nearly equal strength that beat each other by a few pp. X26's
recent pool (14–20 snapshots covering the last ≈ 80–100M) already spans most of that cluster's STEP range. Whether it
spans the cluster's STRATEGIES was the open question for LG-0: §2.4 says it holds 0.90 of the archive's Nash mass, and
that "the cluster" is wider than the support: P\* is within noise of 0.5 against 11 of the 14 nodes it was measured on
from 130M to 270M.

### 2.2 What population levers have done here before (MEASURED, old lineage)

- **The exploiter loop (population loop rounds 1–2, `ai_v13_*`, 2026-09-24/25).** A plateau parent trained against its
  own fresh offense exploiters at share 0.40, against a control at share 0.
  - Round 1: a fresh best responder found a gap 10 pp SMALLER on the loop (Δ = −10.00 pp [−16.60, −3.27]).
  - Round 2: Δ2 = −8.25 pp [−15.01, −1.38].
  - Both intervals exclude 0, but neither clears the registered 5.0 pp bar, so both read NOT DETECTED.
  - The kill guards were quiet: untaught −1.31 pp [−4.19, +1.19], SmallRL +1.75 [−2.13, +5.62].

  So the loop REDUCED EXPLOITABILITY with no measurable cost and no measurable outside gain. It is the one population
  lever in this project with a positive measured effect (ledger L21329, L21387, L21391; `UNDERSTANDING.md` era
  close-out). The inputs were pre-`b0a28b5b`, so this gives a DIRECTION, not a level.
- **PFSP over a homogeneous recent pool** was CONVICTED at −26 to −33 Elo in the pre-era gen-17 attribution
  (`design_compute_efficiency.md` F15; `design_learner_recipe.md` §3 row 5). That is the reason A2 is X35's
  "plausible null".
- **Exploiters are counterplay, not skill:** distilled as TEACHERS they robbed the generalist (`UNDERSTANDING.md` TL;DR
  6, 10, 16). As OPPONENTS they were safe. Every lever below uses exploiters as OPPONENTS only.

### 2.3 What X35 can and cannot detect (ARITHMETIC; FINDING F-AS-1)

X35's R1 classifies Δ_pop by a Welch interval over seed finals at 97.5 % per look (§5.6.4) against a ±2.0 pp bar. With
σ the per-final SD of the population score S (unknown; G1 HOLDs above 1.2 pp):

| look | seeds per side | σ = 0.8 pp | σ = 1.0 pp | σ = 1.2 pp |
|---|---|---|---|---|
| 1 | 3 vs 3 (df 4) | GAIN needs Δ̂ > 4.3 | > 4.9 | > 5.4 |
| 2 | 5 vs 5 (df 8) | GAIN needs Δ̂ > 3.4; NULL needs \|Δ̂\| < 0.6 | > 3.7; < 0.26 | > 4.1; **NULL unreachable** |

Two consequences:
- **An archive or PFSP effect the size of the measured cycling (≈ 0–3 pp) cannot read GAIN.** It reads UNDECIDED with
  Δ̂ > 0, or, if σ is small, NULL.
- **R1 is effectively a two-way test, GAIN or UNDECIDED, unless σ̂ ≤ 1.0.** The branches in §7 treat "UNDECIDED,
  Δ̂ > 0" as an outcome with its own action, never as "no effect".

R2 (the best-response gap) is the sensitive meter for exploiter arms. Its Newcombe interval detects a fall of
≈ 7 pp at n ≈ 400 per cell, and round 1 read −10.0.

### 2.4 The league's HEADROOM (LG-0 / X55, MEASURED 2026-10-20)

The read is [`x55_league_headroom_2026-10-20/`](../research_state/measurements/x55_league_headroom_2026-10-20/README.md): part 1 on existing games (the
ladder above and X26's in-run sentinel rows), part 2 P\*'s own row, 18 cells, 5,000 games on `main.h2h` (CPU, rule 5b;
7 cells at 200 pairs, 11 at 100, F-X55-1 there). Its predictions for X35 were registered and pushed BEFORE any game
(`798c4f2a`).
- **The Nash is in C0's pool.** Posterior Nash over the archive to 268M: 0.90 of the mass on C0's 14 seeded
  snapshots (164–268M), 0.98 at ≥ 130M; support 6 (194, 224, 244, 246, 254, 268M), posterior 7.5 [5, 10].
- **A1's archive barely touches it:** ±5M neighbourhoods of A1's nodes (30, 60, 100, 130, 160, 190, 230, 260M) hold
  0.19 (the early four 0.014); R's A-block's 0.21.
- **P\*'s row is flat:** 0.61 / 0.57 / 0.56 against 30 / 60 / 100M; 0.45–0.59 against every node from 130M to 270M,
  no cell below 0.5 at 95 % (worst 244M, 0.450 [0.384, 0.516]); against the Nash mixture (six measured support
  snapshots, 0.74 of the mass) **0.514 ± 0.014**. **S(P\*) = 0.569 ± 0.009**, the score every X35 production arm
  starts from. The ladder's chained spine overstates long-range gaps (0.733 predicted against 30M, 0.61 measured).
- **PFSP has nothing to tilt:** stable PFSP over A1's nodes at the measured rates gives 0.85–1.10× uniform (max / min
  1.3, 3.7 % of the slice moves); A2's factor over C0's pool 0.97–1.05× (1.2 %). No A1 node is near the 0.80 mastery
  floor, so all eight stay live (and cost collection, F-AS-3).
- **Reading.** The archive is a wide flat top that P\* already sits in; an archive or PFSP lever has no measured
  weakness to repair, so its value, if any, is generalisation from more diverse opponents, which nothing here
  measures. The place a population lever can still move is EXPLOITABILITY against a fresh best responder (§2.2,
  LG-1), which this read does not bound.

---

## 3. AlphaStar's components, mapped onto what we have

| AlphaStar component | ours today | flag / code | gap to AlphaStar's form |
|---|---|---|---|
| **Main agent(s)**, one per race | ONE trainee per run, over a team pool; the closest analogue to a race is a team ARCHETYPE (offense / balance / stall) | — | no per-archetype main agents. The team axis is D1 / X41 / X42's (`design_diversity_teams_opponents.md`) |
| **Self-play slice (35 %)** | the recent snapshot pool, 20 deep, sliding window (oldest out), promotion-gated | `snapshot_pool.py`, `--self-play` | ours is ≈ 100 % of the challenge bucket. AlphaStar's 35 % is against the CURRENT self, ours is against frozen recent snapshots |
| **The archive, never pruned** | none in a run. The pool evicts at 20. `--stable-opponents` adds a FIXED set chosen at launch | `fixed_opponent_pool.py`; X35-**A1** (8 evenly spaced nodes, 50 % of self-play) | a set that GROWS in a run is not built. Live-opponent fan-out costs collection (§4.3) |
| **PFSP over all past players (50 %)**, f_hard | (a) pool PFSP: `weight = recency × (1 + S·(1 − p))`, with only `--n-sentinels` snapshots measured per cycle; (b) stable PFSP: weight `1 − p`, floored at 0.05, over un-mastered stable opponents | `--pfsp-scale`, `--n-sentinels` (X35-**A2**); `--stable-opponent-pfsp` | (a) is a mild linear tilt over a HOMOGENEOUS pool, the form our history convicts. (b) is f_hard with p = 1 and could run over A1's archive TODAY. X35 has no arm for it (→ LG-2) |
| **Forgotten-player slice (15 %)** | mastery moves a stable opponent to the bot FLOOR at ≥ 0.80 for 2 cycles; pool spread retention keeps the oldest | `--stable-opponent-mastered-wr`, `--pool-spread` | ours retires what is mastered, AlphaStar re-targets what is forgotten (§1.1) |
| **Payoff matrix over every pair** | the eval ledger (counts, append-only), the cycle monitor's thinned matrix (offline), `win_rate_vs_ext_<label>` per stable opponent per cycle, and PFSP's EMA over the measured sentinels | `main.eval_ledger`, `main.cycle_monitor`, `SelfPlayCallback` | in-trainer coverage is partial (5–10 sentinels plus the stable set); the T20 pool matrix is not built |
| **Main exploiters** (vs the current main agents, reset) | one ROUND: 3 exploiters of a FROZEN P\*, 5M each, folded in at share 0.4 | `--exploiter`, `--no-self-play`, the two-phase X35 kit (**A3**) | ours target a frozen P\* and are not reset or re-run (F24). Two learners on one GPU are not designed, so rounds must be TIME-SLICED (§4.2) |
| **League exploiters** (PFSP vs the whole league, reset) | 3 exploiters vs P\* + A1's archive at share 1.0 with stable PFSP | X35-**A6** (two-phase kit) | conditional in X35. Same frozen-round form |
| **Supervised init + KL to the human policy** | none: training from scratch. Human replays are X50 (owner Q1, ai_v11 punted 2026-08-18) | — | out of scope here. Pokémon bots reach the top without it (§1.5) |
| **Evaluation against exploiters** | `main.best_response_gap` (a fresh reader per target), X35's R2 / G-E | built | readers are 5M offense-archetype runs. Unpinned readers need T58 |

---

## 4. Each component: requirement, expected value, what is built, the cheapest experiment

### 4.1 The archive (keep past players)

- **What it requires:** frozen nodes as opponents (BUILT), a choice of WHICH nodes, and for a growing archive, a way
  to add nodes during a run under a cap on live opponents.
- **Expected value given our cycling.** It is bounded by the cyclic component the recent pool does NOT already cover.
  A1 picks 8 nodes evenly over [0, P\* − 20M]: 30, 60, 100, 130, 160, 190, 230, 260M for P\* = 280M (the frozen R
  file). The Nash cluster (§2.1) sits in C0's own pool, and LG-0 (§2.4) measured P\* at only 0.56–0.61 against A1's
  early nodes and ≈ 0.49–0.53 against its late ones: coverage against FORGETTING of a flat archive, not coverage of
  the CLUSTER. JUDGMENT: Δ_pop ≈ 0–2 pp; registered: +0.5 pp, 80 % [−1.0, +2.0] (X55 `PREDICTIONS.md` P2).
- **Built:** A1 (flags), the pool seed's step filter (T54), the X35 kit. The cycle monitor reads the archive's matrix
  offline.
- **Cheapest experiment:** A1 itself (batch 1, queued). Before it reads, LG-0 on CPU: which nodes carry the Nash
  weight, and how much A1's nodes overlap them.

### 4.2 Exploiters (main and league)

- **What it requires:** a best-response run against a frozen target (BUILT: `--exploiter --no-self-play`, 5M ≈ 1.1
  GPU-h), a fold-in as stable opponents (BUILT), and a reader (BUILT: `main.best_response_gap`). AlphaStar's form
  also RE-TARGETS the moving main agent and RESETS the exploiter. On one GPU that means time-slicing: pause the main
  agent, train an exploiter of its latest snapshot, resume with it in the opponent set. Nothing schedules that today.
- **Expected value.** On exploitability it is LARGE and measured: −8 to −10 pp of gap per round on the old lineage.
  On the population score it is unknown and plausibly small. On outside meters none was detected. It is the only
  lever whose mechanism (finding holes outside the cluster) is not bounded by the cyclic size of §2.1.
- **Built:** A3 and A6 as two-phase kits (T55); the G-E readers registered. **R0 (AM-R0, 2026-10-20):** one best
  responder per side (`toy_r0` against T\*, `r0` against P\*), +20M each, read as an exploitability CURVE every 2M
  ([`R0_REGISTRATION.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md)).
  **R0-toy is READ (2026-10-21):** headroom H **+11.8 pp [+9.1, +14.5]** against T\* (predicted +22 [+10, +34]), most
  of it by 4M (t90 4M), class RISING only through a 10–12M dip (P(SATURATED) 0.60 missed). **R0-prod is READ
  (2026-10-22):** H **+11.1 pp [+8.3, +13.8]** against P\* (predicted +18 [+8, +30]; learned part +13.6 over gap(0)
  −2.5), class **SATURATED** (D −0.3 [−4.2, +3.6]), t90 8M; all 11 points inside their bands. LG-1's 5M statistic
  (in-run rows) **+6.0 pp** (112 / 200): **A6 is NOT triggered** (bar 0.15). A 5M round reads only ≈ 2/3 of P\*'s
  hole (+7 at 4M); the toy and production holes are the same size (+0.8 [−3.1, +4.6]).
- **Cheapest experiment:** A3 with R0 (LG-1). The cheaper-exploiter literature (§1.2: minimax exploiter) is a
  COST-REDUCER for LG-3's repeated rounds, not a first step.

### 4.3 PFSP and the payoff matrix

- **What it requires:** a win rate for every candidate opponent, and a weighting. AlphaStar's f_hard concentrates
  games on the opponents the agent loses to. Its premise is a DIVERSE league.
- **Expected value.** Over a recent, homogeneous pool it is ≈ 0 or negative (our history, F15). Over an archive it
  would concentrate the archive slice on the nodes that still beat P\*, i.e. SELECT the cluster automatically, IF any
  node beat P\*. **LG-0 (§2.4) found none:** over A1's archive f_hard's weights span only 0.85–1.10× uniform (max / min
  1.3), so LG-2 ≈ A1 plus noise unless the archive or P\* changes. JUDGMENT: over today's archive, ≈ A1.
- **Built:** `--stable-opponent-pfsp` (weight `1 − p`, floored at 0.05) with `win_rate_vs_ext_<label>` measured
  every cycle for every stable opponent. A1's 8 nodes are each evaluated every 2M (100 games), so the payoff row PFSP
  needs EXISTS for the archive.
- **The fan-out cost (MEASURED, a throughput hazard for any league).** Collection at 256 envs ran 5,453 env steps/s
  with 5 pool snapshots and 3,507 with 20 (`design_compute_efficiency.md` §1.6). Every live opponent is one more
  small forward per host step. A league of 30+ live members would cut collection further, unless the GROUPED FORWARD
  (`--t2-grouped-opponents`, X45(b), BUILT, GPU effect UNMEASURED) lands or a live-opponent cap (X45(a), k of N per
  rollout, rotated) is built. A1 adds 8 live slots to the pool's ≈ 14–20. Its extra wall time is UNMEASURED; the
  X35 sheet declares only ≈ 2 % for the extra eval games (FINDING F-AS-3).
- **Cheapest experiment:** LG-2, A1 + `--stable-opponent-pfsp`, one lever over A1. No build.

### 4.4 A league that grows and keeps running (the composed form)

- **What it requires:**
  - B-ARCH (≈ 1 agent-day): add a frozen node to the stable set every Δ steps (e.g. 10M), with a cap K on LIVE
    stable opponents (sample K of N per rollout, PFSP-weighted, rotated). This is the "CPU-resident league with a
    PFSP-chosen GPU active set" that `design_league_decisions.md` §B already names.
  - B-ROUND (≈ 1.5–2 agent-days): a launcher-level schedule that, every M steps, pauses the main run, trains R reset
    exploiters of its newest snapshot (5M each), and resumes with their finals added to the archive. The restart
    machinery and the two-phase kit exist; the loop does not.
  - The existing meters, run on the long continuation: Tier 1, the cycle monitor, the population score against the
    frozen R, and R2 per round.
- **Expected value:** the sum of what its parts read. It is the AlphaStar deliverable, and it is ADOPTED only as a
  composition of parts that each read ADOPT-CANDIDATE, or UNDECIDED with Δ̂ > 0 together with a falling gap.
- **Cheapest experiment:** on the TOY (≈ 3× cheaper per step, and its plateau T\* exists), and only after X35's
  transfer calibration says the toy transfers for population levers (A1 / A2 agree in class), §5.2 there.

### 4.5 Equilibrium-seeking single-agent training (the league-free alternative)

- **What it requires:** a KL term toward a trailing "magnet" copy of the policy (R-NaD / MMD). That is X35-A4, T56, ≈ 2
  agent-days, ≈ +3 % of an update.
- **Expected value.** It targets the same oscillation as the archive, without extra opponents or fan-out. DeepNash is
  the existence proof (§1.4), and MMD measured the PPO-with-a-magnet form on small imperfect-information games; at our size it is UNVERIFIED.
- **Cheapest experiment:** A4 as registered, in a batch whose pin carries T56.

---

## 5. What we will NOT build now, and why

| option | why not now | what would change that |
|---|---|---|
| **A faithful AlphaStar league** (12 concurrent learners, exploiters on 10⁹-step timeouts) | one GPU cannot host concurrent learners, and TStarBot-X shows a 1/30-scale copy collapsing (§1.2) | never at this compute. LG-3 is the time-sliced, cheap-exploiter form |
| **PSRO with a Nash meta-solver as the deliverable** | PSRO's answer is a MIXTURE of policies. We deploy one network and its search, so the population must be folded back (AlphaStar's main agent). The meta-solver's role is covered by PFSP (LG-2) and, if ever needed, A5's Nash weights (T57) | if search could act as a mixture over networks, which is not in scope (one-ply search only) |
| **Pipeline PSRO** | its gain is parallel best responses, which one GPU cannot run | a multi-GPU box |
| **XDO / NXDO** | it needs an infostate-indexed restricted game; ours has far too many infostates and no such interface | never in this design |
| **NeuPL / Simplex-NeuPL** | a new conditioning input (the opponent mixture σ) means a new architecture and a fresh lineage, before we know whether a population helps at all | LG-3 reads ADOPT-CANDIDATE AND the live-opponent fan-out binds even with the grouped forward. Then a conditional population is the way to hold the league in one forward |
| **AlphaStar's supervised init + KL to a human policy** | there is no human policy (X50 S0 census first; owner Q1 on re-opening ai_v11) | the owner re-opens human data. VGC-Bench's behaviour-cloning gain (§1.5) is the argument for it |
| **AlphaStar Unplugged-style offline RL** | that is Metamon's route, and our advantage is the simulator | — |
| **X35-A5 (Nash-weighted static archive) before LG-2** | it needs T57 (≈ 2 agent-days), and on a near-transitive archive its static Nash is near-pure (F-ED-3), so it will often refuse (support < 3). LG-2's PFSP adapts online at no build | LG-0 shows a wide Nash support over A1's nodes |
| **Diversity bonuses** (PSD-PSRO style) | Yao et al. show that diversity metrics do not guarantee a better equilibrium; we would measure exploitability anyway | LG-3 shows exploiters finding the SAME hole every round (no diversity) |

---

## 6. The ranked roadmap (after X35 batch 1)

**The ranking rule** is expected information per GPU-hour toward the question "does a population lever break X26's
plateau?", with built levers ahead of unbuilt levers of equal promise. **Where batch 1 stands:** its toy block T1 (toy
C0 + X47) is on the GPU, and production P1 (A1 / C0, ABBA) waits for FP32_ALT's GPU cost read
(`design_compute_efficiency.md` §5.6, AM-8 / AM-9). On the expected path the first A1-vs-C0 read lands ≈ 26 production
GPU-h after P1 starts. **Every step below needs the orchestrator's GO. GPU-h are production unless marked; the
backlog ids are in `../research_state/EXPERIMENT_BACKLOG.md` (new rows AS-1 to AS-4 = X55–X58; LG-1 and LG-4 are the existing rows CE-3c and CE-3d).**

| rank | id | step | when | GPU-h | agent-days | decision it feeds |
|---|---|---|---|---|---|---|
| 1 | **LG-0** (X55) | **The league HEADROOM read, on CPU.** (a) The max-entropy Nash posterior over X26's nodes from the banked ladder and the monitor rows: which nodes carry weight. (b) How much of that weight A1's 8 archive nodes and R's A-block hold. (c) P\*'s score row against the Nash-support nodes and A1's nodes, topped up to 200 pairs per node on `main.h2h` (≈ 6,400 games, ≈ 1–1.5 h of rule 5b's CPU). (d) **Registered predictions** for A1, A2 and A3 (Δ_pop and R2) written BEFORE A1's look 1 reads | **DONE 2026-10-20** (§2.4; predictions pushed `798c4f2a` before any X35 production arm existed) | **0** | ≈ 0.5 | whether LG-2 should re-pick its archive from the Nash support; a prior against which batch 1 is read |
| 2 | **LG-1** (X35-A3 + R0, row CE-3c) | **MAIN EXPLOITERS, folded in, with R0.** R0 = a fresh best responder of P\* (exploitability baseline, A6's trigger). **R0 is now X35's registered R0-prod (AM-R0, 2026-10-20): it runs right after P1's C0 seeds, +20M as a curve, and its first 5M is this R0** ([`R0_REGISTRATION.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md)). **READ 2026-10-22: the 5M statistic is +6.0 pp, so A6 is not triggered; H +11.1 pp, SATURATED by 20M, t90 8M.** A3 as registered: 3 exploiters of P\* (5M each), then 3 seed continuations with them at share 0.4 and stable PFSP. R2 is the co-primary. **Run at batch 1's production pin, so it shares C0 (saving ≈ 13 GPU-h)** | the first production arm after batch 1's P1–P4 | ≈ 3.3 + 13 + 1.4 ≈ **18** (R0 is AM-R0's) | ≈ 0.25 (the kit stage) | is the gap-fall of the old lineage real on X26's lineage? It gates LG-3's exploiter rounds and A6 |
| 3 | **LG-2** (X56) | **PFSP OVER THE ARCHIVE**, AlphaStar's main-agent PFSP slice: A1's argv + `--stable-opponent-pfsp` (f_hard with p = 1 over the un-mastered archive). One lever over A1. R1 against C0, and Δ against A1 reported. If LG-0 shows A1's nodes miss the Nash support, a SECOND arm picks the archive from the support instead (the archive choice as its own lever). **LG-0 (§2.4): A1's nodes DO miss it (0.19 of the mass), but the support is C0's own recent pool (0.90), so a re-picked arm is C0 with a reweighted pool: the second arm is MOOT; and f_hard over A1's archive is near-uniform (max / min 1.3), so LG-2 ≈ A1 on today's archive. Its rank is the orchestrator's call (§8)**. **AM-RP-b (2026-10-20): the second arm is DROPPED, and LG-2 runs only on an A1 GAIN** | only on an A1 GAIN (AM-RP-b) | ≈ **14.4** | ≈ 0.25 (a kit arm) | does hardness-weighting over a diverse archive move what uniform sampling did not? It supersedes X35-A5 in rank (A5 needs T57 and its static Nash on a near-transitive archive degenerates, F-ED-3) |
| 4 | **LG-3** (X57) | **THE LEAGUE-LITE CONTINUATION, toy first.** From T\*: a growing archive (a node every 10M, live cap K = 8 of N, PFSP-weighted, rotated per rollout; B-ARCH), PFSP over it, and every 50M a time-sliced round of 3 reset exploiters of the newest node (5M each; B-ROUND), their finals joining the archive. Run +200M against a matched plain continuation of +200M. Read: Tier 1 and the cycle monitor on both, the population score against the frozen toy R at every 50M, R2 per round. **Production repeats it only if the toy reads ADOPT-CANDIDATE, or UNDECIDED with Δ̂ > 0 and the gap falling, AND X35's transfer calibration did not read a disagreement for population levers** | gated: (LG-1 R2 falls) OR (A1 or LG-2 reads GAIN, or UNDECIDED with Δ̂ > 0) | toy ≈ 12 ON + 12 control + ≈ 3.5 exploiters + readers ≈ **30–35 (toy)**; production at +200M ≈ **105** | B-ARCH ≈ 1, B-ROUND ≈ 1.5–2, read plan ≈ 0.5: **≈ 3–4** | THE AlphaStar-style deliverable: does the composed league keep climbing where the plain continuation is flat? |
| 5 | **LG-4** (X35-A4, row CE-3d) | **The TRAILING MAGNET** (R-NaD / MMD): β · KL(π ‖ π_mag), β 0.05, magnet refreshed every 2M. The league-free alternative: equilibrium-seeking with no extra opponents and no fan-out | in the first batch whose pin carries T56 | ≈ **15** | T56 ≈ **2** | **Moves to rank 3** if A1 reads NULL / LOSS and LG-1's gap does not fall (no population headroom, so a league has nothing to cover) |
| 6 | **LG-5** (X58) | **THE MULTI-LINEAGE league**, CONDITIONAL. P\* trains with T\*'s archive and the closing-test seeds' finals (`rb_es_end2_s2001`…`s2008`, other lineages at HEAD's architecture) as stable opponents. Production and the toy are AlphaStar's "several main agents". R must then be re-frozen with an X-block NOT trained against. X29's basin probe (a fresh generalist against the incumbent pool) is the deeper form | after LG-3 on the toy, iff the toy league reads ADOPT-CANDIDATE or the cross-lineage cells of R stay in band while the in-lineage ones saturate | ≈ **15** (3-seed fork); X29 ≈ 30–60 | ≈ **1** (a dry-run of cross-architecture stable opponents in the TRAINER, F-AS-5; the R re-freeze) | is the plateau a property of one lineage's basin? |

**Total on the expected path** (LG-0, LG-1, LG-2, LG-3 on the toy): ≈ 65–70 GPU-h and ≈ 4.5–5 agent-days. LG-3 on
production adds ≈ 105 GPU-h, LG-4 ≈ 15 and LG-5 ≈ 15. For scale, batch 1's expected path is ≈ 91 GPU-h
(`design_compute_efficiency.md` §5.6, AM-7).

**What is NOT re-ranked here.** A6 (league exploiters) keeps X35's trigger (A3 ADOPT-CANDIDATE or gap(P\*) ≥ 0.15 at
R0). LG-1 supplies both inputs. A7 stays trigger-gated. D1 / X38 and the team-axis levers (X41, X42) keep their own
ranks in `design_diversity_teams_opponents.md`, but §7 says how D1's result reorders this list.

---

## 7. Decision branches on X35's results

Each row is a batch-1 (or LG-1) outcome and what it changes in §6. Classes are X35's (§5.6.4). Where an outcome is
UNDECIDED, the branch uses the sign of Δ̂ only to ORDER work, never to adopt (F-AS-1).

| outcome | reading | change to the roadmap |
|---|---|---|
| **G1 HOLDs** (σ̂ > 1.2 pp) | seed noise of S is too large for a 2 pp bar | every later arm (LG-1, LG-2, LG-4) is registered at 5 seeds per side from the start, about ×1.7 the GPU-h; LG-3's single long continuation stays descriptive only |
| **A1 GAIN** | forgetting or cycle coverage was binding, and larger than the measured cyclic size predicts | LG-2 next (PFSP over the archive), then LG-3 with B-ARCH as its first build. A5 drops (LG-2 adapts the weights online). LG-4 drops to "complement" |
| **A1 UNDECIDED, Δ̂ > 0** (the modal case, F-AS-1) | consistent with a 0–3 pp effect | LG-2 does NOT run (AM-RP-b: LG-2 only on an A1 GAIN); LG-3 is gated on LG-1's R2. The A1 report says "no effect ≥ the R1 bar detected", never "the archive does not help" |
| **A1 UNDECIDED, Δ̂ ≤ 0, or NULL, or LOSS** | the archive is not where the plateau is | LG-2 is DROPPED (no archive to weight). LG-4 moves to rank 3. LG-3 keeps only its exploiter rounds (B-ROUND), if LG-1's gap falls |
| **A2** | DROPPED from batch 1 (AM-RP-c, 2026-10-20): its lever tilts 1.2 % of games, NULL is unreachable, its N_prom is confounded and its PFSP state biased (F31) | none: A2 is not run, so its two rows (GAIN ⇒ LG-2 to rank 2; NULL / UNDECIDED ⇒ no change) cannot fire |
| **D1 GAIN on the unfamiliar panels** | the plateau is in TEAM coverage, not policy coverage | the league's diversity should come from teams: X42 (off-meta exploiters, the team-axis main exploiter) runs in LG-1's slot or beside it, and LG-3's exploiter rounds use team-pinned exploiters |
| **X47 GAIN / PER-STEP ONLY** (toy depth helps) | capacity binds at the plateau | X47b (production depth 3 → 4) goes ahead of LG-2. Population levers on a network that cannot represent more strategies are second |
| **Transfer calibration: A1 / A2 classes AGREE** (not expected: AM-RP-a accepts NOT SHOWN now) | the toy may pre-screen population levers | LG-3 runs on the toy first, as planned, and the toy may RANK LG-2-style variants |
| **Transfer calibration: DISAGREE / NOT SHOWN** (**APPLIED IN ADVANCE**, AM-RP-a) | population levers do not transfer, or the read could not show it | LG-3's toy run becomes descriptive only, and production LG-3 needs its own gate (an ADOPT-CANDIDATE from LG-1 or LG-2 on production) |
| **R0 (AM-R0's R0-prod, its first 5M): gap(P\*) ≥ 0.15** | P\* is clearly exploitable | A6 is triggered (X35's rule), and LG-3's exploiter rounds rank above its archive build. R0's CURVE adds the headroom H and whether it saturates by 20M: a RISING curve says a 5M exploiter round under-reads the hole, so LG-3's rounds need more than 5M each |
| **LG-1: R2 gap FALLS, G-F passes, R1 not LOSS** (A3's verdict, AM-RP-a: a **ROBUSTNESS GAIN**, never "stronger") | the old lineage's pattern: less exploitable, not measurably stronger against R | LG-3 goes ahead (the league's purpose at a plateau is robustness first). The G-K Kakuna read decides whether robustness shows outside |
| **LG-1: R2 does not fall** | exploiters do not reduce X26-lineage exploitability | B-ROUND is dropped from LG-3. If A1 is also not positive, the league family is DEPRIORITISED: LG-4, X47b and search (`search_ledger.md`) go first |

---

## FINDINGS (standing rule 7)

- **F-AS-1. X35's R1 cannot read GAIN below ≈ 3.4–5.4 pp, and cannot read NULL at σ ≥ 1.0 pp.** This is arithmetic
  (§2.3) on the registered rule: a Welch interval at 97.5 % with 3 vs 3 seeds (df 4) or 5 vs 5 (df 8), against a
  ±2.0 pp bar. An archive or PFSP lever whose effect is about the measured cyclic size (≈ 3 pp) will most likely read
  UNDECIDED. X35's docs state such a bound for X47 ("4.6 pp") but not for R1. Every branch in §7 treats an UNDECIDED
  as an outcome with its own action.
- **F-AS-2. A1's evenly spaced archive tests coverage against FORGETTING, not coverage of the plateau's Nash cluster.**
  A1's nodes are 30, 60, 100, 130, 160, 190, 230, 260M (the frozen R file; "20–230M" here was the stand-in P\* = 180M
  spacing, corrected 2026-10-20). The X35 doc already notes that R's A-block sits next to A1's nodes (F20). **LG-0
  MEASURED the overlap (§2.4):** the Nash mass within ±5M of A1's nodes is 0.19 (early four 0.014), against 0.90 in
  C0's own pool; and P\* measures 0.56–0.61 against A1's early nodes, so they are not "opponents P\* already beats"
  by much either.
- **F-AS-3. A1's extra live opponents cost collection throughput, and the cost is UNMEASURED and undeclared.**
  Collection fell from 5,453 to 3,507 env steps/s going from 5 to 20 pool snapshots. A1 adds 8 live slots, and the
  sheet declares only ≈ 2 % for the extra eval games. R1 compares at matched STEPS, so the strength read is clean.
  Any per-GPU-hour claim, and LG-3's cost, must record `gpu_wait_ms_per_host_step`. **ESTIMATED by LG-0 and recorded
  in the X35 sheet as F-LS-25:** +7.6 to +9.8 % of wall time at equal steps (X26's own +1.20 ms per snapshot per host
  step, or the bench's +1.55, × 8 live slots × ≈ 516 host steps per 65.6 s iteration), ≈ 4.7–4.8 GPU-h per A1 seed;
  not measured on CPU (a CPU forward costs per row, the GPU per-slot path per call). A1 s3501's first 1M measures it.
- **F-AS-4. X35 has no arm for PFSP over the ARCHIVE**, which is AlphaStar's form. A2 tests PFSP over the recent
  homogeneous pool, the form our history convicts (F15). LG-2 is that missing arm, and it needs no build.
- **F-AS-5. Cross-architecture stable opponents in the TRAINER are UNVERIFIED.** X26 and the toy have the same
  `arch_signature` (`gen3_mon_tied_gain_v1`, config 158, read from both runs' `model_config.json`), so
  `check_opponent_compatible` admits a toy node as a production run's stable opponent. Its docstring assumes that the
  same signature means the same network sizes, and that is false for the toy. Whether the opponent loader and the
  inference service build the toy at its own sizes is UNVERIFIED. LG-5 must dry-run and smoke it first. `main.h2h`'s
  two-architecture engine is a separate, verified path.
- **F-AS-6. §2.1's numbers come from the in-run snapshot ladder** (50–100 pairs per edge, edges chosen by the pool),
  not the registered cycle-monitor instrument. They are descriptive.
- **F-AS-7. The population-loop evidence (§2.2) is from the PRE-`b0a28b5b` lineage**, so it gives a direction, not a
  level. LG-1 is its replication on the current lineage.
- **F-AS-8. Sharing batch 1's C0 assumes LG-1 and LG-2 launch at batch 1's production pin. VERIFIED 2026-10-20**
  ([`STAGED_PLAN.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md) §4.3). The pin and
  the pool record are per SIDE, so a new arm at the written pin shares its side's C0 and needs no +13 GPU-h. But the kit
  must first register the arm: `kit.ARMS` / `build_argv`, a side-pinned arm set that `pin_check`, `stage_status`'s AM-8
  gate, `k9b_batch` and `disk_check` read, and `readout`'s arm labels. That is a separate unit of ≈ 0.25–0.5
  agent-day.
- **F-AS-9. Literature items that are UNVERIFIED** (§1): AlphaStar's PFSP exponent p, its payoff estimator, and its
  exploiter / snapshot step counts (the exponents are garbled in the manuscript PDF; the values come from TStarBot-X
  and DI-engine); ROA-Star's arXiv / OpenReview id; the compute of AlphaStar Unplugged, DeepNash and Pipeline PSRO's
  Barrage run; NeuPL's compute as a number; Metamon's self-play effect as a number; the Hernandez et al. CoG 2019
  text; Bansal et al.'s exact column mapping; the venues of Spinning Tops and Policy-Space Diversity from their own
  pages. Nature's page and DOI were not checked against Nature itself, only through the manuscript.
- **F-AS-10. The brief's NeuPL author list was wrong** (Piliouras and Gemp are not authors). It is corrected in §1.3.
  The AlphaStar main-exploiter fact that `design_compute_efficiency.md` F24 marks "UNVERIFIED, from memory" (they
  track the moving main agent) is now VERIFIED (§1.1). **That doc's F23 / F24 now cite §1.1 (edited by LG-0,
  2026-10-20).**
- **F-AS-11. X35-A3's exploiters use the f_var-free form.** AlphaStar's main exploiter switches to f_var over
  main-agent-derived players when its win rate is below 20 %. Ours trains only against the frozen P\*. At 5M steps
  from a P\* fork the exploiter starts at about 50 %, so the switch would not fire. The difference matters only for
  LG-3's reset rounds, where an exploiter reset to the SUPERVISED parameters (AlphaStar) has no analogue: we reset to
  the newest main node (SCC's branching).

---

## 8. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-20 | **This doc CREATED** (research + design worker, orchestrator brief for goal 4) | "AlphaStar-style" defined for our compute as FOUR mechanisms (archive, PFSP over it, reset exploiters folded in, the main agent's mix), each mapped onto built flags and X35's arms; the expected value bounded by the MEASURED cycling (≈ 3 pp in the last 80M, ≈ 4 pp over the archive, Nash support ≈ 8); exploitability named as the meter where population levers move | a faithful AlphaStar league (no concurrent learners; TStarBot-X's scale-down collapse); PSRO / P2SRO / XDO as the deliverable; NeuPL now | §1 (two literature workers, primary sources, 2026-10-20); §2 (`cycle_monitor_retro_2026-10-19/`, ledger L21329 / L21387 / L21391) |
| 2026-10-20 | **The ranked roadmap after X35 batch 1** (same worker; PROPOSED, needs the orchestrator's GO per step) | LG-0 headroom read (CPU, now) → LG-1 main exploiters + R0 (X35-A3, at batch 1's pin, sharing C0) → LG-2 PFSP over the archive (A1 + `--stable-opponent-pfsp`, no build) → LG-3 the league-lite continuation on the toy first (B-ARCH + B-ROUND, ≈ 3–4 agent-days) → LG-4 the trailing magnet (A4) → LG-5 multi-lineage, conditional | A1 / A2 variants first (the cyclic size ≈ 0–3 pp sits below R1's GAIN bar, F-AS-1); A5 before LG-2 (a build, and a near-pure Nash); exploiters as teachers (the old lineage's robbery) | §6, §7; AlphaStar Fig. 3 (exploiters the largest component); population loop rounds 1–2 (gap −10.0 / −8.25 pp) |
| 2026-10-20 | **Decision branches registered BEFORE any X35 production read** (same worker) | §7's table: what each class of G1, A1, A2, D1, X47, the transfer calibration, R0 and LG-1's R2 does to the order; an UNDECIDED orders work by the sign of Δ̂ but adopts nothing | branching on head-to-head results (one opponent, ruled out in a cyclic regime, `design_league_decisions.md` §0) | §2.3's arithmetic; X35 §5.6.4 |
| 2026-10-20 | **LG-0 (X55) READ** (X55 worker; predictions FIRST, `798c4f2a`, then the read) | §2.4 added: the archive's Nash is in C0's recent pool (0.90), A1's nodes' neighbourhoods hold 0.19, P\*'s measured row is flat (0.45–0.61 over 18 cells, none below 0.5 at 95 %; 0.514 ± 0.014 against the Nash mixture), S(P\*) = 0.569, PFSP near-uniform over A1's archive (max / min 1.3). §0, §2.1, §4.1, §4.3, the LG-0 / LG-2 rows of §6, F-AS-2 (A1's real nodes 30–260M), F-AS-3 (estimated cost) and F-AS-10 updated. LG-2's "second arm" is MOOT (the support is C0's pool). **No rank is changed here**: whether LG-2 drops below LG-4 on this read is the orchestrator's call | leaving LG-2's premise ("PFSP selects the cluster automatically") standing after the read showed no node beats P\* | [`x55_league_headroom_2026-10-20/`](../research_state/measurements/x55_league_headroom_2026-10-20/README.md) |
| 2026-10-20 | **AM-RP: the orchestrator's rulings on the X35 REPLAN memo** (registration worker; `BATCH1_LAUNCH_SHEET.md` §0 AM-RP) | LG-2 runs only on an A1 GAIN and its second arm is dropped; A2 is dropped from batch 1; the transfer calibration is accepted as NOT SHOWN now, so LG-3 on the toy is descriptive; A3's verdict is "R2 falls AND G-F passes AND R1 not LOSS", labelled ROBUSTNESS GAIN. §6's LG-2 row and §7's A1 / A2 / calibration / LG-1 rows changed | LG-2 on an A1 UNDECIDED with Δ̂ > 0 (f_hard over A1's archive is near-uniform, max / min 1.3, so LG-2 ≈ A1); A2 deferred rather than dropped | [`REPLAN_2026-10-20.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/REPLAN_2026-10-20.md) §1, §4; §2.4 |
| 2026-10-20 | **F-AS-8 VERIFIED** (the AM-RP registration worker, CPU, read-only) | a new arm CAN share its side's C0 at a written pin (the pin and the pool record are per side); the kit needs the arm registered first (≈ 0.25–0.5 agent-day) | each batch-2 arm with its own C0 (+ ≈ 13 GPU-h per side) | `STAGED_PLAN.md` §4.3; `batch2_sprt/f_as_8_kit_dryform.txt` |
| 2026-10-20 | **AM-R0: R0 registered as an exploitability CURVE on both sides** (registration worker; launch sheet §0 AM-R0, the owner's direction "train an exploiter … against prod to see how much head room we have") | R0-toy (T\*, toy pin, right after the GPU battery) and R0-prod (P\*, production pin, right after P1's C0 seeds), +20M each, read every 2M; R0-prod's first 5M is LG-1's R0, so LG-1 drops to ≈ 18 GPU-h; "exploiter + pi search" registered as DESCRIPTIVE cells needing a live search player | a 5M R0 only (no saturation read); a full-pool best responder (no meter) | [`R0_REGISTRATION.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md) |
