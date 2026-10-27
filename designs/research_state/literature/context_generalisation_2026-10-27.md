# Memorised teams, a blind critic, and what the literature says about generalising across contexts (as of 2026-10-27)

> A dated literature survey from web research on 2026-10-27, a read of the local Metamon checkout (`0a00a759`) and of
> Wang 2024 (`designs/references/wang2024_pokemon_rl.pdf`). It is a SNAPSHOT and is not updated; a later survey gets a
> new dated file. The always-current views are `../UNDERSTANDING.md`, `../../endstate/design_beat_metamon_path.md` and
> `../../endstate/search_ledger.md`. It builds on [`sota_survey_2026-10-26.md`](sota_survey_2026-10-26.md) (the
> N1–N5 levers) and does not repeat it. This survey asks a narrower question that the last four days of reads opened.
>
> **Evidence tags.** **VERIFIED** = the claim was read in the primary source (an arXiv abstract, HTML or PDF, a
> proceedings page, the author's repository, or the local Metamon checkout), during this unit. "(abstract)" limits the
> verification to the abstract. **UNVERIFIED** = from a search snippet, a secondary source, or a source that would not
> open. **MY JUDGEMENT** = an inference no source states. Venues marked "(venue from memory)" were not on the page that
> was read. Every number about OUR stack names the read that owns it; this file holds no measurement of its own.
>
> **Abbreviations.** pp = percentage points. P\* = X26's plateau checkpoint (`rb_x26_s2601` @ 280M). U = the pooled
> score on the three unfamiliar panels (H = `heldout72_hpfix`, F = `forum72`, P72 = procedural). home72 = 72 of the
> 719 pool teams. V = P\*'s critic. Context = what a level is in Procgen and a team is for us: the fixed part of an
> episode that the agent can memorise. PLR = prioritised level replay. UED = unsupervised environment design. BC =
> behaviour cloning. OOD = out of distribution. AUC = area under the ROC curve.

---

## 0. The answer, in plain language

**Our finding is the textbook one.** P\* trains on 719 fixed teams and pilots them about 11 pp better than unfamiliar
ones. Every RL study that varies the number of training contexts finds this regime below about 10⁴ contexts:
near-perfect play on the training contexts and a large gap on new ones (CoinRun, Procgen, Zhang et al.). With a
finite set of contexts, the reward-maximising policy specialises to each context, in Bertran et al.'s words
"instance-specific speed-running". SMACv2 found the same in StarCraft micro: fixed unit compositions let a policy that
sees only the clock win. A memorised team plan would be our version of that.

**Whether the edge goes with the whole team or with the sets is OPEN.** The brief's premise ("the premium travels
with the whole team, not the set") came from the forensics. The adversarial review that landed during this unit shows
that the per-mon measure behind it restates the game result and cannot tell team from set (`90708bff`,
`../measurements/h_mid_value_test_2026-10-26/REVIEW.md` B1). rc72b is the discriminating read. This survey therefore
treats team-level memorisation (H-CTX) as a hypothesis, and the recommendations below say what changes under each
answer.

**Recombining familiar parts is supported, with a condition.** The only clean tests of "new combinations of familiar
parts" are outside games:
- CompoSuite: held-out combinations succeed 0.88 when the agent trained on 224 of 256 combinations, and 0.08–0.18 on
  56.
- Hill et al.: systematic generalisation grows with the number of distinct items seen.
- SMACv2: randomised mixes of familiar unit types.
- Kirk et al.'s survey: "combinatorial interpolation" is easier than extrapolation.

The condition is that each part must have been seen in MANY contexts; a part seen in one context does not transfer.
X62's corpus meets that condition by construction (§6.1). No source tests it in Pokémon.

**Breadth has a per-context price, and the field pays it.** As training contexts grow, play on any one training
context falls:
- VGC-Bench: 1 → 64 teams.
- CoinRun: train score 99.5 → 89.2.
- Hill et al.: train accuracy 1.00 → 0.91.

D1's in-pool −1.6 pp is this price, not a regression.

**The critic's blind spot is expected, and calibration tools will not fix it alone.** Value functions overfit
training contexts more than policies do (Raileanu and Fergus: agents trained on more levels end with a HIGHER value
loss and generalise better). The field's fixes all assume the deficit appears in the value's own training target:
- decoupled or partly decoupled critics (IDAAC, PPG, Nafi et al.);
- ensembles (LEEP, randomised priors);
- conservative values (CQL).

Ours does not appear there. V is trained on self-play outcomes, so it cannot learn a premium that exists only
against a non-self opponent. The concurrent review makes the same point (REVIEW M1), and finds that the third of the
gap V "sees" is a turn-1 team prior carried mostly by P72 (M2). No calibration method can learn a premium that is
absent from V's targets. The fixes that change V's targets are breadth and non-self opponents.

**Search on a well-ranked value should be invariant to the level error, but the literature gives it little OOD
lift.**
- For: AlphaZero-family search normalises Q within the tree (MuZero's min–max) or needs only the order of actions
  (Gumbel MuZero). A level shift shared by all of one node's successors therefore cancels.
- Against: where test-time search was measured OOD, it helped slightly or not at all. Hamrick et al.: "Planning alone
  is insufficient to drive strong generalization". In Brick Tic-Tac-Toe, more MCTS did not generalise but training
  on more configurations did. KataGo loses > 97 % to adversarial policies despite search.
- One positive case: a theorem prover where test-time MCTS solved 21 % more OOD problems.

That fits S-U's own registered guess (DOES-NOT-CLOSE-B likeliest).

**Pokémon and loadout games measure less than we do.**
- **Pokémon.** VGC-Bench is the only Pokémon paper that measures piloting unseen teams, at ≤ 64 training teams.
  Metamon gets breadth from 10⁴–10⁵ teams but reports no held-out-team test, and its README says it "has overfit" to
  its own evaluation team set. Random-battle bots avoid the question by drawing a fresh team every game.
- **Loadout games.** OpenAI Five, Honor of Kings, Hearthstone and Legends of Code and Magic all train on random draws
  from the whole loadout space. None holds out lineups or decks built from familiar parts.
- **The one recipe that matches our situation:** Honor of Kings trained specialists on fixed lineups, distilled them,
  then CONTINUED on random lineups from the whole pool. That is P\* plus a long broad continuation.

**The ranked recommendations are in §7.** In short:
1. X62 is the best-posed next lever, with its in-pool cost pre-registered as the expected price.
2. Breadth needs dose: the field's results come from whole runs on the broad distribution, not 20M forks.
3. Do not build a critic-calibration lever.
4. Keep S-U cost-capped.
5. A toy context-count dose curve is the measurement the literature lacks for our domain.

---

## 1. The facts the question rests on

| quantity | value | owner |
|---|---|---|
| familiar vs unfamiliar, P\* vs Kakuna | home72 0.598, U 0.487 | X51; `../../STATUS.md` §1 |
| where the gap forms | from materially LEVEL positions after turn 10: level-at-10 conversion 0.609 home72 vs 0.501 U (+10.8 pp [+4.8, +17.0]); state at turn 10 carries −0.4 pp of 11.2 | `../measurements/piloting_gap_forensics_2026-10-26/` (`c758fcb8`, descriptive) |
| team or set | the forensics read "team, not set" from a per-mon KO drop of +0.093 [+0.004, +0.176]. **The review finds that measure restates the result**: the within-team seen-vs-unseen contrast on U is −0.007 [−0.093, +0.070], which cannot separate the accounts. **OPEN until rc72b** | forensics (`c758fcb8`); REVIEW B1 (`90708bff`) |
| the critic | Δbias +0.076 [+0.016, +0.139]; AUC 0.663 home72 vs 0.667 U (ranking intact); at equal V, U wins 8.6 pp less. Review: a self-play critic cannot learn an opponent-relative premium (M1), and V's 3.5 pp set gap is a turn-1 prior, mostly P72's (M2) | H-MID (`f200258f`); REVIEW M1–M2 |
| set-level identity | COMPOSITIONAL: a familiar set's re-draw moves P\* 1.19× [1.07, 1.33] an unfamiliar one's (bar 1.5×); N2 dropped | N2 (`fd4160d7`) |
| plasticity | NO-LOSS: P\*'s refit AUC 0.986× @25M's; a neutral target refits 1.11× slower, all of it paid by 175M | plasticity check (`8bebdfe3`) |
| breadth so far | D1 (1,981 forum teams, 0.5 share, 20M): U +2.10 [−0.92, +5.11], in-pool −1.61 [−2.90, −0.32]; D3 look 1 +1.27 [−2.63, +5.16]; N1-P (12,000 procedural) being read | `../../STATUS.md` §3 |
| recombination | X62 DRAFT: 12,000 new sixes of verbatim pool sets, ≤ 2 shared with any pool team; D1's corpus is already 31 % recombined-like, N1-P's 0.6 % | `../../endstate/design_x62_team_recombination.md` (`fca58b24`) |
| corpus dose | at share 0.5 over 20M a corpus team is piloted ≈ 24 times a side; a pool team ≈ 395 | X62 design §2 |
| architecture | one shared trunk; the critic reads `value_pooled` through `win_head`, the actor its own MLP (partial separation) | `../../ARCHITECTURE.md` |

---

## 2. Question 1: memorising training contexts vs generalising

| source | claim (with the number) | bearing on our levers | confidence |
|---|---|---|---|
| **Cobbe et al., "Quantifying Generalization in RL"** (CoinRun), ICML 2019, arXiv 1812.02341. VERIFIED (PDF, App. E Table 1) | PPO, 256M steps. IMPALA-CNN train / test: 100 levels 99.4 / 66.6; 500 99.2 / 80.3; 4,000 97.7 / 95.9; 16,000 98.0 / 97.8; unbounded 98.1 / 98.3. Nature-CNN's TRAIN score falls 99.5 → 89.2 as levels rise. "substantial overfitting occurs when there are less than 4,000 training levels" (§3.2). Bigger networks generalise better at every size; L2, dropout, augmentation and batch norm each help a little, and together only "slightly better" | 719 teams sits where every curve shows a large gap. 12,000 (N1-P, X62) sits where CoinRun's gap is ≤ 2 points. The train-score drop is D1's in-pool cost | HIGH for the shape; the level→team mapping is MY JUDGEMENT |
| **Cobbe et al., Procgen**, ICML 2020, arXiv 1912.01588. VERIFIED (PDF §3.1, §4) | 16 games, 100 to 100,000 levels, 200M steps: "To close the generalization gap, agents need access to as many as 10,000 levels"; past a threshold TRAIN performance also rises (an implicit curriculum); IMPALA ×2 / ×4 width improves both | the 10⁴ target behind N1 and X62's 12,000; width = N4 | HIGH (shape), MEDIUM (transfer to teams) |
| **Zhang, Vinyals, Munos, Bengio, "A Study on Overfitting in Deep RL"**, arXiv 1804.06893 (2018). VERIFIED (PDF §4) | Agents memorise thousands of levels, even under random rewards. The number "enough" is task-dependent: ≈ 100 levels for simple mazes, > 1,000 for harder ones, with identical optimal train reward | the threshold for Gen 3 OU is unknown and may be higher than Procgen's (a team space is far richer than a maze); our in-house meters, like train reward, cannot see the gap (X37b) | HIGH |
| **Packer et al., "Assessing Generalization in Deep RL"**, arXiv 1810.12282 (2018). VERIFIED (PDF) | PPO default 78.2 / interpolation 70.6 / extrapolation 48.4; "extrapolation is more difficult than interpolation"; plain PPO / A2C beat specialised generalisation methods | a corpus must COVER the test distribution: X62 covers F's familiar-set slots (70 %), not H's or P72's unfamiliar sets (62–69 %) | MEDIUM (classic control) |
| **Ghosh et al., "Why Generalization in RL is Difficult: Epistemic POMDPs"**, arXiv 2107.06277 (NeurIPS 2021, venue from memory). VERIFIED (PDF) | Generalising from finitely many contexts "induces implicit partial observability"; a policy optimal on every training context can be arbitrarily bad at test (Remark 5.2). LEEP (4 PPO policies on resampled level sets, pulled together) beats PPO on 3 of 4 Procgen games at 200 levels | the formal account of H-CTX: P\* is optimal for "which of 719 teams am I", which is the wrong question on a new team; ensembles are one fix, breadth the other | HIGH (theory), MEDIUM (LEEP's size) |
| **Bertran et al., "Instance-based Generalization in RL"**, NeurIPS 2020, arXiv 2011.01089. VERIFIED (PDF) | Reusing a finite instance set rewards "instance-specific speed-running policies instead of generalizable ones"; the train–test value gap is bounded by the instance count; fix: an instance-agnostic policy ensemble (IAPE) | same as Ghosh, from a different angle; IAPE's cost (several policies) is the same objection | MEDIUM |
| **Kirk et al., "A Survey of Zero-shot Generalisation in Deep RL"**, JAIR 76 (2023), arXiv 2111.09794. VERIFIED (PDF §4.2–4.3) | Defines **combinatorial interpolation** (each factor value seen, the combination not) as harder than ordered interpolation and easier than categorical extrapolation; "Increasing the number of contexts … also makes an evaluation protocol easier" | X62 trains for combinatorial interpolation of sets into teams; N1-P's unseen sets are extrapolation at the set level | HIGH (as a taxonomy) |
| **Hill et al., "Environmental drivers of systematicity and generalisation in a situated agent"**, ICLR 2020, arXiv 1910.00571. VERIFIED (PDF §4.1) | Held-out negation accuracy by the number of words trained with negation: 6 → 0.40, 40 → 0.60, 100 → 0.78, while TRAIN accuracy falls 1.00 → 0.97 → 0.91. Systematicity "can grow with the variety of words/objects experienced" | recombination generalises as the number of distinct contexts per part grows; expect train (pool) play to fall as it does | MEDIUM-HIGH |
| **Mendez et al., CompoSuite**, CoLLAs 2022, arXiv 2207.04136. VERIFIED (PDF Table 2; spot-checked in this unit) | 256 robot tasks = 4 arms × 4 objects × 4 obstacles × 4 objectives. Zero-shot success on held-out COMBINATIONS: multi-task agent 0.18 (56 tasks) → 0.41 (224); a modular compositional agent 0.08 → **0.88**. When an element is seen in only one training task, both fail (0.09 / 0.01). "this capability requires observing a large portion of the tasks" | **the closest test of X62's premise anywhere.** Recombination works with breadth, and better with an architecture that is modular in the parts (our per-mon tokens and attention are partly that, MY JUDGEMENT) | HIGH for the result, MEDIUM for transfer |
| **Ellis et al., SMACv2**, NeurIPS 2023 D&B, arXiv 2212.07489. VERIFIED (ar5iv §5–7, App. E) | In SMAC, "an open-loop policy conditioned only on the timestep can achieve non-trivial win rates"; with randomised unit types (always three familiar types per race) and start positions it "cannot learn a good policy on any maps"; composition drives difficulty | fixed loadouts let a policy store per-loadout scripts. P\*'s team-tied edge is our analogue (MY JUDGEMENT); randomising compositions of familiar parts is what X62 does | MEDIUM-HIGH |
| **PLR** (Jiang et al., arXiv 2010.03934, ICML 2021 from memory), partly VERIFIED (abstract + summarised HTML); **Robust PLR** (Jiang et al., NeurIPS 2021, arXiv 2110.02439), VERIFIED (abstract); **ACCEL** (arXiv 2203.01302) and **PAIRED** (NeurIPS 2020, arXiv 2012.02096), VERIFIED (abstracts) | PLR + UCB-DrAC: "a 76% improvement in mean test return" over uniform sampling at 200 Procgen levels; Robust PLR: not updating on uncurated levels improves robustness; ACCEL: curricula that "start simple but become increasingly complex" | a curriculum adds to breadth, it does not replace it: no source shows a curriculum on ≈ 700 contexts matching ≈ 10⁴ (N3 stays behind N1 / X62) | MEDIUM |
| **Tobin et al., Domain Randomization**, arXiv 1703.06907 (IROS 2017 from memory), abstract via a summariser; **OpenAI, ADR / Rubik's Cube**, arXiv 1910.07113, partly VERIFIED (summarised HTML) | ADR: real-robot median successes 13 → 42 as randomisation entropy rises (Table 3); "increased ADR entropy corresponds to vastly improved sim2real transfer" | more variety transfers better; but randomisation is over a known parameter space, like N1-P's procedural sets | MEDIUM (different domain) |
| **XLand, "Open-Ended Learning Leads to Generally Capable Agents"**, arXiv 2107.12808, partly VERIFIED (summarised HTML) | an agent trained on a vast procedurally generated task space scores on every humanly solvable held-out task; no task-count ablation found | breadth at the extreme; no dose curve | LOW (for our question) |
| **Farebrother et al., "Generalization and Regularization in DQN"**, arXiv 1810.00123. VERIFIED (abstract) | DQN "overspecializes to the training environment"; dropout and L2 help | regularisers help a little; N2's set-channel noise was already dropped by our own test | MEDIUM |

**Reading for question 1.**
1. **How many contexts?** Every controlled study puts the gap's close between 10³ and 10⁴ contexts at 10⁸ steps:
   - CoinRun: overfitting below 4,000 levels, ≈ 2 points left at 16,000.
   - Procgen: "as many as 10,000".
   - Zhang et al.: 10² to > 10³, by task difficulty.

   A Gen 3 OU team is a far richer context than a maze level, so 10⁴ is a floor rather than a target (MY JUDGEMENT).
   12,000 is the right order for a first test.
2. **Does recombination help?** Yes, conditionally. CompoSuite and Hill show held-out combinations of familiar parts
   generalise once the parts have appeared in many combinations, and fail when a part was seen in one. SMACv2 shows
   that randomising compositions of familiar parts removes the memorised script. Kirk places this case between easy
   and hard.
3. **What breadth costs.** Train-context performance falls with breadth in CoinRun (Nature-CNN), Hill and VGC-Bench,
   and recovers only once there is enough breadth (Procgen's "implicit curriculum"). An in-pool drop is the expected
   sign of the lever working, not a reason to stop it.

---

## 3. Question 2: value miscalibration under shift, and search on a level-shifted value

| source | claim | bearing | confidence |
|---|---|---|---|
| **Raileanu and Fergus, "Decoupling Value and Policy for Generalization in RL"** (IDAAC), ICML 2021, arXiv 2102.10330. VERIFIED (abstract spot-checked in this unit; ar5iv §5.4–5.5) | "more information is needed to accurately estimate the value function than to learn the optimal policy. Consequently, the use of a shared representation … can lead to overfitting". Agents trained on more levels end with a HIGHER value loss and generalise better (§5.4); a 200-level agent's value is near-linear in the episode step, i.e. it memorises level length. IDAAC: normalised Procgen test 163.7 vs PPO 78.0 at 25M (Table 1) | the critic is the part that memorises contexts first; our trunk is shared with the critic (only the heads separate) | HIGH (Procgen); its size on our network is unknown |
| **Cobbe et al., Phasic Policy Gradient**, ICML 2021, arXiv 2009.04416. VERIFIED (abstract) | separate policy and value phases avoid interference; auxiliary value distillation keeps shared features | a lighter decoupling than IDAAC | MEDIUM (Procgen test numbers not read) |
| **Nafi, Ali, Hsu, "Analyzing the Sensitivity to Policy-Value Decoupling in Deep RL Generalization"**, NeurIPS 2022 Deep RL Workshop. VERIFIED (abstract) | over 16 Procgen games, "partial separation can easily capture the necessary policy-value representation asymmetry" | we already have partial separation (separate heads on one trunk); how much of the trunk is shared is the open knob | LOW-MEDIUM (workshop) |
| **Ghosh et al.** (LEEP, above); **Lakshminarayanan et al., Deep Ensembles**, NIPS 2017, arXiv 1612.01474; **Osband et al., Randomized Prior Functions**, NeurIPS 2018, arXiv 1806.03335; **An et al., SAC-N / EDAC**, NeurIPS 2021, arXiv 2110.01548. All VERIFIED (abstracts) | ensembles "express higher uncertainty on out-of-distribution examples"; bootstrap ensembles "have no mechanism for uncertainty that does not come from the observed data" (hence random priors); a min over a large Q ensemble penalises OOD actions | an ensemble flags a NEW TEAM as uncertain; it does not know which way the error goes. Our error has a sign (the premium is missing), so an ensemble could at best down-weight V on U | MEDIUM |
| **Kumar et al., CQL**, NeurIPS 2020, arXiv 2006.04779. VERIFIED (abstract) | "overestimation of values induced by the distributional shift"; CQL lower-bounds the value | the same direction as V on U relative to home72 (Δbias > 0), but an offline-RL fix | LOW for us (analogy only) |
| distributional critics and held-out-level generalisation; game-AI win-probability calibration studies | **NOT FOUND** as primary studies (a search, not an exhaustive one) | our "level wrong, ranking intact" reading has no direct precedent in the literature we found | — |
| **Schrittwieser et al., MuZero**, Nature 2020, arXiv 1911.08265. VERIFIED (ar5iv, pUCT appendix) | "MuZero computes normalized Q values … by using the minimum-maximum values observed in the search tree" | an additive level error shared by a node's successors cancels in selection | HIGH (mechanism) |
| **Danihelka et al., "Policy Improvement by Planning with Gumbel"**, ICLR 2022. VERIFIED (ICLR page and a full-text copy) | the root improvement guarantee holds "for any monotonically increasing transformation" of Q; unvisited actions are completed with v_π | search needs only the ORDER of a node's actions; but the scale of σ(q) still sets how far search overrides the prior, so a compressed range on U changes the balance | HIGH (mechanism) |
| **Grill et al., "MCTS as Regularized Policy Optimization"**, ICML 2020, arXiv 2007.12509. VERIFIED (abstract) | AlphaZero's search approximates a KL-regularised policy optimisation over Q | a constant added to every Q leaves the solution unchanged (MY JUDGEMENT from the form); our piKL search (λ·log π) has the same shape | MEDIUM |
| **Hamrick et al., "On the Role of Planning in Model-based Deep RL"**, ICLR 2021, arXiv 2011.04021. VERIFIED (abstract, ar5iv) | planning helps mostly in TRAINING (policy targets, data); at test time only "very slight gains" OOD; "Planning alone is insufficient to drive strong generalization"; depth-2 trees often suffice | test-time search is a weak lever for an OOD gap | HIGH (as stated); our games differ |
| **Anand et al., "Procedural Generalization by Planning with Self-Supervised World Models"**, ICLR 2022, arXiv 2111.01587. VERIFIED (abstract, ar5iv) | Procgen mean-normalised test: MuZero 0.50 vs Q-learning 0.36 vs Q-learning + model 0.45; three factors: "planning, self-supervised representation learning, and procedural data diversity" | planning helps when it is part of TRAINING; data diversity is one of the three | MEDIUM |
| **Tan and Motani, "Brick Tic-Tac-Toe"**, arXiv 2207.05991. VERIFIED (ar5iv; venue unconfirmed) | "merely increasing MCTS lookahead iterations was insufficient for AlphaZero to generalize"; training on more brick configurations reached 62.4 % and 90.4 % | the cleanest small-scale contrast: breadth beats search for an unseen configuration | MEDIUM (toy game) |
| **Wu et al., INT**, ICLR 2021, arXiv 2007.02924. VERIFIED (ar5iv §4.4) | test-time MCTS "helped to solve 21% more problems on average" on theorem distributions outside training | the positive case: search can help OOD where the value still ranks well | MEDIUM |
| **Wang et al., "Adversarial Policies Beat Superhuman Go AIs"**, ICML 2023, arXiv 2211.00241. VERIFIED (abstract) | "&gt;97% win rate against KataGo running at superhuman settings" | search does not rescue a value that misjudges a class of positions | HIGH (as stated) |
| **Tamassia and Böhmer, "Improving Robustness of AlphaZero Algorithms to Test-Time Environment Changes"**, arXiv 2509.04317. VERIFIED (abstract) | simple search changes improve AlphaZero under environment shift, especially at small budgets | a hint that search's settings matter under shift; no numbers read | LOW |
| **ReBeL** (NeurIPS 2020, arXiv 2007.13544), **Student of Games** (Sci. Adv. 2023, arXiv 2112.03178). VERIFIED (abstracts) | sound equilibrium search in imperfect-information games | no OOD evidence; the 10-20 review already placed them outside our regime | — |
| **Wang 2024**, MIT MEng thesis (local PDF). VERIFIED (pp. 30–31, via the 10-26 survey) | MCTS + NN beats NN 0.809, which the author calls "somewhat 'inflated'"; every game is a new random team | search paid IN distribution (random battles have no unfamiliar team) | MEDIUM |

**Reading for question 2.**
1. **Critic overfitting to contexts is a known effect, larger than the policy's** (IDAAC). But our particular miss is
   not the IDAAC kind. IDAAC's value memorised level features that ARE in its target (episode length). P\*'s V is
   trained on self-play outcomes and never sees the familiarity premium, which appears only against a non-self
   opponent (H-MID FINDING 2; X37b; REVIEW M1, which calls the verdict "near predetermined" for this reason). So no
   calibration fix (ensembles, temperature, distributional heads) can teach V a premium its targets do not carry.
   What changes V's targets is training on more teams, or against opponents that are not P\*.
2. **Search on V: the invariance is real, the OOD payoff is thin.** One-ply search compares one node's actions, and
   AlphaZero-family selection is invariant to a shared level shift (MuZero min–max, Gumbel's monotone transform). So
   H-MID's level error does not by itself rule out search on U (H-MID § Reading 3). But where test-time search was
   measured OOD, it helped slightly (Hamrick), not at all (Brick Tic-Tac-Toe), or not against a value blind spot
   (KataGo). The one clear positive (INT, +21 %) is a domain where the value's ranking is the whole problem. H-MID's
   AUC is measured ACROSS positions; search needs the ranking of SIBLING actions on U, which nobody has measured
   (FINDING F5).
3. **The training-time role of search** (Hamrick, Anand) is the version with evidence behind it. It is not one of our
   levers (search is paused, `search_ledger.md`), and nothing here argues to reopen it for goal (b).

---

## 4. Question 3: Pokémon systems and team diversity

| system | how it handles team diversity in training | unseen-team generalisation reported? | confidence |
|---|---|---|---|
| **Metamon** (Grigsby et al., RLC 2025, arXiv 2504.04395; local README `0a00a759`). VERIFIED | offline RL on replay-reconstructed teams; self-play on the procedural Variety set ("11k intentionally diverse teams per gen/tier", paper); team sets "are randomly sampled between episodes" (README l. 857); later sets of 107k (`gl_05_26`) and 22k (`hl_05_26`) Gen 3 teams | **NO held-out-team test.** README l. 865: the `competitive` set was "the main evaluation and deployment set, Metamon has overfit to them". l. 359: Alakazam is "more robust to diverse team choices thanks to a larger dataset" (no number). Paper Limitations: "nonsensical decisions … particularly when the opponent is playing with a rare team" | HIGH (what it does), the generalisation claim is ASSUMED, not measured |
| **VGC-Bench** (Angliss et al., arXiv 2506.10326 v3; AAMAS 2026 from the authors' bib). VERIFIED (§5.2, Tables 3–4) | 1 / 4 / 16 / 64 nested tournament teams; 5,013,504 steps a run, 5 seeds | **YES**, on 72 OOD teams: 64 beats 1 0.669, 64 beats 16 0.564, 16 beats 4 0.547, 4 beats 1 0.595 (±0.03). On seen teams the order reverses (1 beats 4 0.699). "the performance … on any one team decreases considerably and consistently" | HIGH; scale ≤ 64 teams, 1–2 % of our steps |
| **PokéAgent Challenge** (Karten et al., arXiv 2603.15563). VERIFIED | a released 200K+ team dataset; 4thLesson's self-play on ≈ 800k replays of `modern_replays`; PA-Agent selected deployment teams by tournament (> 60 %); Team Q's biggest jump came from the v0 → v1 (curriculum) team set | **NO**; the organisers name it "a substantial generalization challenge" (§3.3) without measuring it | HIGH (as stated); participants' entries are self-reports |
| **PokéChamp** (Karten, Nguyen, Jin, arXiv 2503.04094). VERIFIED (§5.3) | no training; "custom teams" in Gen 9 OU | NO; "team composition significantly impacts performance" | HIGH |
| **Wang 2024** (MIT thesis, local). VERIFIED | a fresh random team every game (Gen 4 random battles) | not applicable; §5.3.3 on team formats: "A robust strategy here would have to involve some team randomization" | HIGH |
| **Sarantinos, "Teamwork under extreme uncertainty"**, arXiv 2212.13338. VERIFIED (PDF) | Gen 7 random battles; team building only proposed | NO; "it is significantly harder for players to master every possible random team than to master a few carefully selected" (§2) | MEDIUM |
| **Reis, Reis, Lau, "VGC AI Competition"**, IEEE CoG 2021. VERIFIED (PDF); CoG 2025 page VERIFIED | random team configurations for "generalized training"; the Battle Track draws new teams every battle | NO results on it | MEDIUM |
| **Porimon** (Zhuo et al., arXiv 2609.32544, Sept 2026). VERIFIED (abstract spot-checked in this unit; HTML read) | an LLM agent; Gen 8 random battles chosen "to reduce the influence of teams" | NO | MEDIUM |
| **Foul Play** (github `pmariglia/foul-play`). Partly VERIFIED (README) | a search bot; no training; its set database is known only from the PokéAgent appendix | NO | LOW for our question |
| **Huang and Lee 2019**, IEEE CoG 2019 (DOI 10.1109/CIG.2019.8848014). **UNVERIFIED** (paywalled; via Sarantinos §3) | a second agent "trained to play with a few selected teams" beat the first against bots but "performed worse … against human players" | an early few-teams-vs-breadth anecdote, no numbers | LOW |
| Simões et al. 2020; Rodriguez et al. 2024 (Springer); Rossi et al. SBGames 2024 | **UNVERIFIED / not relevant**: not opened, or the abstract says nothing about teams | — | — |

**Reading for question 3.**
1. **VGC-Bench is the only Pokémon dose–response.** It is monotone in the team count from 1 to 64 on unseen teams, with
   a per-team cost on seen teams. Nothing measures 10³–10⁵ teams against a held-out set in singles OU. Our X51 meter
   (0.598 vs 0.487) is already more measurement than the field publishes.
2. **Metamon's breadth is in the data, and it admits overfitting to its own evaluation set.** 19 of that set's 20
   Gen 3 teams are in our pool (the 10-26 survey §3.1, ledger 2026-10-18). Its generalisation is assumed from breadth.
3. **Every random-battle system sidesteps the question.** A fresh team every game is the extreme of breadth.
4. **The specialist–generalist trade recurs:**
   - VGC-Bench Table 3 (seen teams);
   - Metamon's Alakazam against Kadabra;
   - Huang and Lee (UNVERIFIED);
   - PA-Agent's selected deployment teams.

---

## 5. Question 4: teams as context in loadout games

| source | loadout handling | unseen-loadout result | confidence |
|---|---|---|---|
| **OpenAI Five** (Berner et al., arXiv 1912.06680). VERIFIED (App. P) | 17 of 117 heroes, random 5-hero lineups per team per game | NO held-out lineup. Pool-size experiment (5 / 17 / 80 heroes, early training): "Training with 80 heroes has a speedup factor of approximately 0.8", with the caveat that it "could … become worse later in training" | HIGH (as stated) |
| **Ye et al., "Towards Playing Full MOBA Games with Deep RL"** (Honor of Kings), NeurIPS 2020, arXiv 2011.12692. VERIFIED (abstract spot-checked; §3.3, Table 1) | 40 heroes (213.6 billion lineups). Randomly presenting lineups from the start "can lead to 'learning collapse'". **Curriculum self-play:** (1) teachers on FIXED lineups, (2) multi-teacher distillation, (3) "continued training by randomly picking lineups in the hero pool". 336 h with it vs > 480 h without, not converged (Table 1) | evaluated on in-pool random lineups only | HIGH for the recipe; a +400 Elo figure read through a summariser is UNVERIFIED |
| **Ye et al., "Mastering Complex Control in MOBA Games"**, AAAI 2020, arXiv 1912.09729. VERIFIED | one agent per hero (8 heroes) | none: specialists | MEDIUM |
| **Chen et al., JueWuDraft**, IEEE ToG 2021, arXiv 2012.10171. VERIFIED (abstract) | drafting by MCTS + a value net | choosing a lineup, not piloting one | LOW (for us) |
| **Xiao et al., "Mastering Strategy Card Game (Hearthstone)"**, arXiv 2303.05197. VERIFIED (ar5iv §II-A) | the agent builds its own 30-card deck every game from ≈ 270 cards per class | no seen-vs-unseen deck split; beat a top-10 Chinese streamer | MEDIUM |
| **Kowalski and Miernik, "Summarizing Strategy Card Game AI Competition"** (Legends of Code and Magic), arXiv 2305.11814. VERIFIED (ar5iv III-C) | with fixed card sets "an agent can achieve amazing results while hardcoding the entire draft phase"; v1.5 "generates a new card set for each match. It forces agents to generalize" | no quantified generalisation; the v1.5 winner ByteRL is "highly exploitable" (Haluska and Schmid, arXiv 2404.16689, VERIFIED abstract) | MEDIUM |
| **Cardsformer** (Xia et al., ECAI 2023, doi 10.3233/FAIA230581). VERIFIED (abstract) | cards encoded from their text | "outperforms the state-of-the-art … even when there are untrained cards in the deck" | MEDIUM (numbers not read) |
| **Bertram et al., "Learning With Generalised Card Representations for MTG"**, IEEE CoG 2024, arXiv 2407.05879. VERIFIED; **Ward, DraftFM**, arXiv 2608.19568 (Aug 2026). VERIFIED (abstract spot-checked in this unit) | cards by features / text, no ids; draft-pick prediction | unseen cards: 55 % human-pick agreement (Bertram), 50.8 / 60.4 / 56.7 % top-1 on held-out expansions (DraftFM) | MEDIUM; drafting, not play |
| **AlphaStar** (Vinyals et al., Nature 2019) | three races, ladder maps | **UNVERIFIED** here (not opened); no held-out-map result found | LOW |
| Q-DeckRec, Gwent, Legends of Runeterra, ad-hoc teamwork | not searched or not found | — | — |

**Reading for question 4.**
1. **No loadout game tests our case.** Our case is training on a fixed set of loadouts, then piloting a held-out
   loadout built from familiar parts. Systems that play many loadouts train on random draws from the whole space (Dota,
   Honor of Kings, Hearthstone, Legends of Code and Magic v1.5).
2. **Fixed loadouts invite scripts.** SMACv2's open-loop policies and Legends of Code and Magic's hard-coded drafts
   both show it, and both were cured by randomising the loadout.
3. **Breadth was cheap where it was measured.** Dota's 80 heroes cost about 20 % more early training than 17 (§5).
   Honor of Kings shows that broad random lineups FROM SCRATCH can collapse, and that specialist teachers plus
   distillation plus a long random-lineup continuation do not. P\* is already a specialist. The literature's next step
   from a specialist is a long continuation on the broad distribution, not a short fork (MY JUDGEMENT from Ye et al.).
4. **Unseen parts (cards) need part-level features** (Cardsformer, Bertram, DraftFM). Our static tokens already read
   move and stat properties, and unseen ENTITIES are not our gap (the 10-26 survey, D0a).

---

## 6. What it means for our levers

### 6.1 X62 (recombination of pool sets into 12,000 new sixes)

- **Its premise is open.** X62 assumes the edge is team-level (H-CTX). The review removed the forensics' evidence for
  that (B1), so rc72b alone decides whether X62's premise holds. The literature below says what to expect IF it holds.
  If rc72b reads set-level instead, the same literature points at set-level novelty: CompoSuite's and Packer's
  coverage arguments apply to sets, which is D3's and N1-P's territory, not X62's.
- **For.**
  - It is the one lever whose shape the compositional literature tests directly (CompoSuite, Hill, SMACv2, Kirk's
    combinatorial interpolation).
  - CompoSuite's failure mode, a part seen in only one context, cannot arise. The corpus copies a pool occurrence
    uniformly, so its 72,000 slots spread over the pool's 4,314 occurrences, ≈ 17 new contexts per occurrence. Even a
    set the pool holds once gets ≈ 17 new sixes (arithmetic on the X62 design's numbers, MY JUDGEMENT on its
    sufficiency).
  - It removes the per-team script SMACv2 found, for the corpus half of the draws.
- **Against.**
  - CompoSuite's MONOLITHIC multi-task agent reached only 0.41 on held-out combinations even at 224 of 256 tasks; the
    modular agent reached 0.88. How modular P\*'s per-mon token path is, is unmeasured.
  - X62 covers F's familiar slots (70 %) far better than H's (38 %) or P72's (31 %), which is extrapolation at the
    set level (Packer, Kirk).
  - The pool still holds half the draws, so its 719 sixes stay memorisable (the design's §8 item 6 names the
    share-1.0 variant).
- **Expected in-pool cost.** Negative and real: VGC-Bench, CoinRun and Hill all show it. Pre-register it as the price,
  not as a harm bar at 0.

### 6.2 N1-P (12,000 procedural teams)

- **For.** It has Procgen's count, and its procedural sets reach H's and P72's unfamiliar-set slots, which X62 cannot.
- **Against.**
  - It changes sets, coherence and context at once (72 % new sets), so a NULL cannot locate the failure.
  - Procedural content helps only where it covers the test distribution (Packer). Metamon calls its own procedural
    leads "unrealistic".
  - The literature's breadth results come from training on the broad distribution FROM THE START (Procgen, CoinRun,
    VGC-Bench) or over a long continuation (Honor of Kings). N1-P's 20M fork gives each team ≈ 24 pilots.
- **What a NULL would mean.** Under this reading, an N1-P NULL is weak evidence against breadth. It is consistent
  with too little dose and the wrong kind of novelty.

### 6.3 S-U (one-ply search on V, on U)

- **For.** Search is invariant to H-MID's level error (§3).
- **Against.** The literature's OOD test-time gains are small or nil, and its positive cases are domains where
  ranking is the whole problem.
- **What is missing.** The sibling-action ranking on U is the unmeasured quantity that decides it (F5).
- **Verdict.** This agrees with S-U's registered guess. Spend its ≈ 1,000 CPU-h only where it is cheap (a rented box).

### 6.4 Value-training breadth

- **What the literature says.** Critics overfit contexts more than policies (IDAAC), and the cure the literature
  agrees on is more contexts. IDAAC's agents with more levels ended with a higher value loss and generalised better.
- **Why calibration tools miss our case.** Our critic's miss is a premium absent from its targets, so breadth, which
  every lever above already brings, is the value lever too. A critic-only lever (ensembles, decoupling) has no
  evidence of moving a TARGET-ABSENT error.
- **Decoupling as a breadth companion.** Decoupling (IDAAC / PPG) is a candidate companion to breadth, not a
  substitute. Its Procgen gains came on top of ≥ 200 levels.

---

## 7. Recommendations, ranked

"Confidence" is how strongly the literature supports the recommendation's premise, not an effect size on our stack.
Every effect is a GUESS until a registered read.

1. **Make X62 the next breadth lever if rc72b supports H-CTX, and register its in-pool cost as the expected price.**
   - **Why.** Recombination of familiar parts is the best-tested route from context memorisation to held-out
     combinations (CompoSuite, Hill, SMACv2). X62 meets their "many contexts per part" condition by construction.
   - **What to change in the DRAFT.**
     - (a) A non-inferiority margin on home72 sized like D1's −1.6 pp and VGC-Bench's per-team cost, not a harm bar at
       0.
     - (b) A per-mon seen-set edge on F as the mechanism read: the forensics' +0.093 should shrink.
     - (c) The share-1.0 trainee variant (§8 item 6) as the named follow-up, because SMACv2 and Procgen say what remains
       memorisable keeps being memorised.
   - **If rc72b KILLS H-CTX** (the edge is set-level), shelve X62 as its design says. The literature's breadth
     argument then transfers to set novelty, and recommendation 2's union corpus carries it through N1-P's and D3's
     sets.
   - **Confidence:** MEDIUM. No Pokémon test, and a monolithic network recombines worse than a modular one.
2. **Put dose before more 20M plateau forks.**
   - **Why.** Every breadth result in §2, §4 and §5 trained on the broad distribution for the whole run, at 10⁸
     steps (Procgen 200M, CoinRun 256M, VGC-Bench's whole run, OpenAI Five). Honor of Kings, the one specialist-first
     recipe, followed its fixed-lineup teachers with a LONG random-lineup continuation.
   - **The next lever after X62 / N1-P's reads.** Not another 20M fork, but a long continuation of P\* (≥ 100M) on
     the UNION corpus, with the pool at ≤ 0.2 of draws:
     - X62's recombined sixes, for F-like teams;
     - N1-P's procedural sets, for H- and P72-like slots;
     - D1's forum teams.
   - **Relation to X53-N1.** It is the cheap cousin of X53-N1's from-scratch run, and its own read decides whether
     X53-N1 is worth its 150–250 GPU-h.
   - **Confidence:** MEDIUM-LOW. The step counts are other domains', and our plasticity check says a plateau fork can
     still learn.
3. **Do not build a critic-calibration lever (ensembles, temperature, a distributional or conservative head) for goal
   (b).**
   - **Why.** The literature's calibration and uncertainty fixes act on errors present in the value's own targets. Our
     premium is absent from V's self-play targets.
   - **If a critic change is tried.** The evidence-backed form is IDAAC / PPG-style decoupling of the trunk, tested on
     the toy WITH a broad corpus (its gains were measured at ≥ 200 levels), never alone on the pool.
   - **Confidence:** MEDIUM for "not a standalone lever"; LOW for decoupling's payoff on our network.
4. **Keep S-U cost-capped at stage 0 unless a rented CPU box makes the full read cheap.**
   - **Why.** Search is invariant to the level error, but test-time OOD gains are small in the literature (Hamrick,
     Brick Tic-Tac-Toe, KataGo).
   - **A cheaper decider.** Before the full read, the banked-position quantity that decides it: V's ranking of
     SIBLING actions on U against home72. For example, the rank correlation of one-ply Q with a deeper rollout, per
     set. It costs CPU-minutes per thousand positions, not a 1,000 CPU-h game read (an ESTIMATE from S-U's per-decision
     cost).
   - **Confidence:** MEDIUM.
5. **Measure our own context-count curve on the toy: 719 / ≈ 3k / 12k recombined teams, from scratch, to 100M.**
   - **Why.** It is the one measurement the literature lacks for team games. VGC-Bench stops at 64 teams, and no
     singles-OU study exists. Zhang et al. show the threshold is task-specific, and CoinRun shows the curve's shape
     decides whether 12,000 is enough.
   - **Cost.** About 3 × 6.3 toy GPU-h plus ≈ 9 lane-h per arm of Kakuna reads (the 10-26 survey's N1 costing,
     ESTIMATE). It shares the N1 / N4 toy frame.
   - **Confidence** that the curve is informative: MEDIUM-HIGH. That the toy's curve transfers to production: LOW (the
     toy's familiarity premium is about half P\*'s, X51).

---

## 8. FINDINGS (hazards, skips, and what could not be verified)

- **F1. The 10-26 survey misstates Kakuna's 63 % as a ladder GXE.** Metamon's README says its GXE table values "have
  no connection to ratings on the public ladder" (`/home/goodlad/dev/metamon/README.md` l. 397, VERIFIED at
  `0a00a759`). They are relative to the listed models. That survey is a snapshot and is not edited; this line is the
  correction.
- **F2. Metamon's Variety set size disagrees between sources.** The paper says "11k … per gen/tier"; the Hugging Face
  team card lists `paper_variety` at about 1k teams for Gens 1–4 (a sub-agent's read; not reconciled).
- **F3. Kakuna's own self-play team sets remain UNVERIFIED** (the 10-26 survey's F3, unchanged).
- **F4. Several primaries were read through a summarising fetch rather than verbatim:** PLR's full text, ADR, XLand,
  Tobin, and the Honor of Kings Elo figure. Their quotes are marked partly VERIFIED or UNVERIFIED in the tables.
  Section-level quotes for IDAAC, Hamrick and Anand come from ar5iv renderings of the arXiv text.
- **F5. "Ranking intact" is measured across positions, not across sibling actions.** H-MID's AUC compares games. Search
  needs a node's actions ranked correctly on U. No read has measured that, and recommendation 4 names it.
- **F6. The reading that V's targets cannot carry the premium** was my inference from H-MID FINDING 2 and X37b. The
  concurrent review (`90708bff`, REVIEW M1) reached it independently. No read has measured V's level in self-play
  games where one side pilots an unfamiliar team.
- **F6b. The brief's premise was overtaken during this unit.** "The premium travels with the whole team, not the set"
  was withdrawn as a finding by REVIEW B1 while this survey was being written (team vs set OPEN until rc72b). The
  survey is written conditional on that read: §0, §1, §6.1 and recommendation 1.
- **F7. Not found:** a distributional-critic generalisation study; a game-AI win-probability calibration study; any
  held-out-lineup or held-out-deck PILOTING result in a loadout game. Searches were not exhaustive.
- **F8. Not opened (UNVERIFIED):** Huang and Lee 2019 (paywalled), Simões et al. 2020, Rodriguez et al. 2024, AlphaStar,
  DeepNash, Ward et al. 2021 (MTG drafting), Bhatt et al. 2018, Vieira et al. 2020.
- **F9. The ≈ 17 contexts per pool occurrence in §6.1 is arithmetic on the X62 design's own counts** (72,000 corpus
  slots over 4,314 pool mons), not a property of a built corpus; X62's corpus is NOT BUILT.
- **F10. Sub-agent reading.** Four sub-agents read the sources in parallel. I re-opened CompoSuite's Table 2, the
  IDAAC, Ye et al. 2020, Porimon and DraftFM abstracts, and the Metamon README lines quoted here. Every other
  VERIFIED tag rests on a sub-agent's read of the primary source.

---

## 9. Sources

General RL generalisation:
- Cobbe et al., CoinRun, https://arxiv.org/abs/1812.02341. Cobbe et al., Procgen, https://arxiv.org/abs/1912.01588.
- Zhang et al., https://arxiv.org/abs/1804.06893. Packer et al., https://arxiv.org/abs/1810.12282.
- Ghosh et al., https://arxiv.org/abs/2107.06277. Bertran et al., https://arxiv.org/abs/2011.01089.
- Kirk et al., https://arxiv.org/abs/2111.09794. Hill et al., https://arxiv.org/abs/1910.00571.
- Mendez et al., CompoSuite, https://arxiv.org/abs/2207.04136. Ellis et al., SMACv2, https://arxiv.org/abs/2212.07489.
- Jiang et al., PLR, https://arxiv.org/abs/2010.03934; Robust PLR, https://arxiv.org/abs/2110.02439.
  Parker-Holder et al., ACCEL, https://arxiv.org/abs/2203.01302. Dennis et al., PAIRED, https://arxiv.org/abs/2012.02096.
- Tobin et al., https://arxiv.org/abs/1703.06907. OpenAI, ADR, https://arxiv.org/abs/1910.07113.
  XLand, https://arxiv.org/abs/2107.12808. Farebrother et al., https://arxiv.org/abs/1810.00123.

Value and search:
- Raileanu and Fergus, https://arxiv.org/abs/2102.10330. Cobbe et al., PPG, https://arxiv.org/abs/2009.04416.
  Nafi et al., https://neurips.cc/virtual/2022/64096.
- Lakshminarayanan et al., https://arxiv.org/abs/1612.01474. Osband et al., https://arxiv.org/abs/1806.03335.
  An et al., https://arxiv.org/abs/2110.01548. Kumar et al., CQL, https://arxiv.org/abs/2006.04779.
- Schrittwieser et al., MuZero, https://arxiv.org/abs/1911.08265. Danihelka et al.,
  https://iclr.cc/virtual/2022/poster/6418. Grill et al., https://arxiv.org/abs/2007.12509.
- Hamrick et al., https://arxiv.org/abs/2011.04021. Anand et al., https://arxiv.org/abs/2111.01587.
  Tan and Motani, https://arxiv.org/abs/2207.05991. Wu et al., INT, https://arxiv.org/abs/2007.02924.
  Wang et al., https://arxiv.org/abs/2211.00241. Tamassia and Böhmer, https://arxiv.org/abs/2509.04317.
- Brown et al., ReBeL, https://arxiv.org/abs/2007.13544. Schmid et al., https://arxiv.org/abs/2112.03178.

Pokémon:
- Grigsby et al., Metamon, https://arxiv.org/abs/2504.04395; local `/home/goodlad/dev/metamon` @ `0a00a759`
  (README l. 359, 397, 857, 865); https://huggingface.co/datasets/jakegrigsby/metamon-teams.
- Angliss et al., VGC-Bench, https://arxiv.org/abs/2506.10326. Karten et al., PokéAgent,
  https://arxiv.org/abs/2603.15563. Karten et al., PokéChamp, https://arxiv.org/abs/2503.04094.
- Wang 2024, `designs/references/wang2024_pokemon_rl.pdf`; https://dspace.mit.edu/handle/1721.1/153888.
- Sarantinos, https://arxiv.org/abs/2212.13338. Reis et al., https://ludii.games/citations/COG2021-2.pdf;
  https://cog2025.inesc-id.pt/vgc/. Zhuo et al., Porimon, https://arxiv.org/abs/2609.32544.
  Foul Play, https://github.com/pmariglia/foul-play.

Loadout games:
- Berner et al., OpenAI Five, https://arxiv.org/abs/1912.06680. Ye et al., https://arxiv.org/abs/2011.12692 and
  https://arxiv.org/abs/1912.09729. Chen et al., JueWuDraft, https://arxiv.org/abs/2012.10171.
- Xiao et al., https://arxiv.org/abs/2303.05197. Kowalski and Miernik, https://arxiv.org/abs/2305.11814.
  Haluska and Schmid, https://arxiv.org/abs/2404.16689. Xia et al., Cardsformer, doi 10.3233/FAIA230581.
  Bertram et al., https://arxiv.org/abs/2407.05879. Ward, DraftFM, https://arxiv.org/abs/2608.19568.

Ours:
- `../measurements/piloting_gap_forensics_2026-10-26/README.md`, `../measurements/h_mid_value_test_2026-10-26/README.md`,
  `../measurements/n2_identity_test_2026-10-26/README.md`, `../measurements/plasticity_check_2026-10-26/README.md`,
  `../measurements/h_mid_value_test_2026-10-26/REVIEW.md` (the adversarial review, `90708bff`).
- `../../endstate/design_x62_team_recombination.md`, `../../endstate/design_search_u_headroom.md`,
  `../../endstate/strategy_if_levers_null_2026-10-26.md`, `../../STATUS.md`, the 10-26 survey.
