# X64: the toy context-count curve. Does the familiarity gap shrink as the number of training teams grows?

**Status: DRAFT (2026-10-27), explicit-only. Nothing is built, registered or launched.** It makes runnable
recommendation 5 of the context-generalisation survey
([`../research_state/literature/context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md)
§7, `5d554c23`): measure our own context-count curve on the toy at 719, ≈ 3k and ≈ 12k teams, each trained fresh.
It borrows its corpus from X62 ([`design_x62_team_recombination.md`](design_x62_team_recombination.md) §2), its toy
recipe and the review of the last fresh-toy design from X61
([`x61_n1_breadth_2026-10-25/REVIEW.md`](../research_state/measurements/x61_n1_breadth_2026-10-25/REVIEW.md),
NOT-READY), and its toy speeds and screen style from N4
([`design_n4_capacity_screen.md`](design_n4_capacity_screen.md)). Its registration is a separate unit (§9).

**Abbreviations.**
- **T\*** is the toy's plateau checkpoint: X34 (`rb_toy_ent_s4002_deep`) at 290,000,087 steps. **P\*** is
  production's: X26 at 280M.
- **The pool** is the 719 teams every toy and P\* trained on. **home72** is 72 of them. **F** = `forum72`,
  **H** = `heldout72_hpfix`, **P72** = the procedural panel. **U** = H, F and P72 at equal weight (goal (b)'s meter).
  **rc72b** is H-CTX's panel: 72 new sixes of pool sets, ≤ 2 shared with any pool team
  ([`hctx_rc72_2026-10-26/PREREG.md`](../research_state/measurements/hctx_rc72_2026-10-26/PREREG.md), Amendment 2).
- **N** is an arm's number of distinct training teams. **x** = log10 N.
- **G** = home72 − U against Kakuna, the generalisation gap (this doc's primary). **G_ctx** = home72 − rc72b, the
  context-only gap (same sets, new six).
- **S** is in-house strength: `main.x35.readout.population_score` against the toy's frozen reference population R
  on home72.
- pp = percentage points. LB / UB = lower / upper bound. **GPU-h** = hours of our card. **lane-h** = one wall-hour
  of the Kakuna anchor lane.
- **GUESS** = an unmeasured effect. **ESTIMATE** = a cost from measured rates. **MY JUDGEMENT** = an inference no
  read states.

---

## 0. One screen

- **The question.** On the toy, trained from step 0 at matched steps, does the gap between our own teams and
  unfamiliar teams shrink as the number of distinct training teams grows from 719 to ≈ 12,700? And if it does, does
  it shrink because unfamiliar teams get better, or because our own teams get worse?
- **Why it is needed.** Every controlled study puts the gap's close between 10³ and 10⁴ contexts (CoinRun, Procgen,
  Zhang et al.). No study measures it for a team game: VGC-Bench stops at 64 teams. Our breadth levers (D1, D3,
  N1-P, X62) are 20M plateau forks, which give each new team ≈ 24 pilots, so a NULL from any of them cannot say
  whether breadth fails or the dose was too small. X64 is the dose-response curve from step 0.
- **The arms. One lever at three doses, all fresh, one pin, matched steps (100M):**

  | arm | N (distinct teams) | the lever | pilots per team per side at 100M (ESTIMATE) |
  |---|---:|---|---:|
  | **A-719** | 719 | none: the fresh toy C0 | ≈ 3,950 |
  | **A-3.7k** | 3,719 | `--team-mix x62_recomb_v1@K3:0.8067 --team-mix-sides both` | ≈ 760 |
  | **A-12.7k** | 12,719 | `--team-mix x62_recomb_v1@ALL:0.9435 --team-mix-sides both` | ≈ 220 |

  - The extra teams are **X62's recombined sixes**: every mon block is copied verbatim from a pool team, so all
    three arms train on the same 658 sets. Only the number of sixes changes (§2.1).
  - The share K / (K + 719) gives every team, pool or corpus, the same draw rate, so N is the count of equally
    weighted contexts, as a Procgen level count is (§2.2).
- **The dose.** 100M is NOT the toy's plateau (X34's Tier 1 read GAIN until 270M). X64 is therefore a
  matched-budget read in the memorisation regime, not a plateau read (§3.1).
- **Seeds and cost.** Recommended: 3 / 2 / 3 seeds (the endpoints carry the slope), 8 runs, **≈ 50 GPU-h**. A budget
  variant at 2 / 1 / 2 costs **≈ 32 GPU-h**. The survey's one seed per arm (≈ 19 GPU-h) cannot estimate seed
  variance and is not a readable design (FINDING 1).
  - CPU: ≈ 84–120 lane-h of Kakuna for the recommended design (≈ 52–75 for the budget variant).
  - Plus the K0 gate (≈ 19–27 lane-h) and ≈ 8–15 lane-h of in-house cells.
- **The read (§4): the slope of G on log10 N, on all finals, one look.** Δ = the fitted change in G from 719 to
  12,719 teams, with a 90 % interval widened for seed variance (floor 2 pp):
  - **SHRINKS** iff UB < 0. It then routes on the sign of the fitted U change:
    - **SHRINKS-U**: U rises, so the gap closes from the unfamiliar side;
    - **SHRINKS-HOME**: U falls, so the gap closes by forgetting the pool.
  - **GROWS** iff LB > 0.
  - **FLAT** iff LB > −2.0 pp, and only when A-719's own gap is detected (LB > 0). Otherwise UNINFORMATIVE.
  - **UNDECIDED** otherwise.
- **Power (§5).** X64 is a large-effect detector. For the recommended design, SHRINKS reads:
  - 0.29–0.36 at a true −3 pp over the range;
  - 0.42–0.59 at −4 pp;
  - 0.67–0.92 at −6 pp.

  A false SHRINKS reads 0.02–0.06. FLAT reads ≈ 0.16–0.20 under a null.
- **What it implies (§6).**
  - **SHRINKS-U:** breadth from step 0 is a lever on the toy. X62 and N1-P are worth production GPU, and a NULL
    from them reads as dose, which routes to a long or from-scratch breadth run (X53-N1, the survey's
    recommendation 2). The 3.7k arm's position says whether ≈ 3k teams already suffice.
  - **FLAT:** context breadth is not the lever on the toy. The from-step-0 breadth runs lose their support.
  - **Otherwise:** nothing is licensed.
- **The gates (§7).**
  - rc72b does not read KILLED;
  - X62's corpus is built with its K3 tier;
  - the CPU K0 gate passes: the old toy seeds at 100M show G ≥ +2.0 pp and G_ctx ≥ +1.0 pp;
  - a PREREG passes adversarial review.
- **Is it a cheaper screen than X62?** No, not on the GPU: ≈ 50 GPU-h against X62's 20.5. It is a cheaper screen
  than the from-scratch or long-dose production breadth runs it routes (≈ 100–250 GPU-h), and on the Kakuna lane,
  the binding resource, it costs about what X62's read does (§7.3).

---

## 1. The question, and what the literature leaves open

**What the literature says.** The survey's §2 found the same shape in every RL study that varies the number of
training contexts. Below ≈ 10³–10⁴ contexts the agent plays its training contexts near perfectly and new ones badly;
past that, the gap closes:
- CoinRun: substantial overfitting below 4,000 levels, ≈ 2 points left at 16,000;
- Procgen: "as many as 10,000 levels";
- Zhang et al.: 10² to > 10³, depending on the task.

Train-context performance falls as contexts grow (CoinRun's Nature-CNN, Hill et al., VGC-Bench). That is the
in-pool price.

**What it does not say.**
- **Where the threshold sits for a Gen 3 OU team.** A team is a far richer context than a maze level (survey §2,
  MY JUDGEMENT), and Zhang et al. show the threshold depends on the task.
- **Whether recombining familiar parts counts as a new context.** CompoSuite and Hill et al. say yes, once each part
  has appeared in many contexts. X62's corpus meets that by construction: ≈ 17 new sixes per pool occurrence (survey
  §6.1, arithmetic).
- **How much dose a new context needs.** Our breadth forks give a corpus team ≈ 24 pilots per side (X62 §2). X52
  needed ≈ 4,000 pilots per team to show a practised-team edge (+3.2 pp [−0.2, +6.8], `42bb0858d`).

**Why the toy, and why now.**
- **The toy is the only place a from-step-0 curve is affordable.** Three production arms to 100M would cost
  ≈ 20 GPU-h per seed at ≈ 1,400 steps/s. The toy runs ≈ 3.3× faster per step.
- **Plateau forks can learn.** The plasticity check read NO-LOSS
  ([`plasticity_check_2026-10-26/README.md`](../research_state/measurements/plasticity_check_2026-10-26/README.md)).
  So a breadth-fork NULL would point at dose or at the wrong kind of breadth, not at a frozen trunk. X64 separates
  the two: it varies the count at a dose ≈ 10× the fork's per corpus team.
- **The toy has a gap to close, but a small one.** T\* drops 3.5 pp from home72 to F, against P\*'s 6.8 (X51,
  `3010c81ba`). Its home72 − U is unmeasured (FINDING 6), and at 100M it is likely smaller still (X61 REVIEW B2). The
  K0 gate (§4.6) measures it before any GPU is spent.

---

## 2. The arms

### 2.1 Which teams: recombined, not procedural

| corpus | what grows with N | sets across the arms | its arm variable | built? |
|---|---|---|---|---|
| **`x62_recomb_v1`** (X62: verbatim pool blocks, ≤ 2 shared with any pool team, pool-matched species profile with Smogon teammate lift) | the number of sixes only | **the pool's 658 sets in every arm's pool half; 572 of them in the recombined part** (X62's STRICT Hidden Power rule; FINDING 3) | context count | NO (≈ 1 agent-day; X62 builds it on its own gate) |
| `n1_proc_v1` (N1-P: 12,000 Smogon-usage teams, tiers K1 ⊂ K3 ⊂ K10 ⊂ ALL) | sixes AND new sets: 72 % of its slots are sets P\* never piloted, 9,372 of 15,655 sets seen once | grows with N | contexts + set novelty + incoherence | yes |

**Chosen: recombined.** The reasons, in order of weight:
1. **The arm variable must be the context count alone.**
   - With `n1_proc_v1`'s prefixes, each larger arm also adds thousands of new, independently drawn sets. A
     slope would be a slope in contexts, set novelty and incoherence at once, which is X62 §1.2's objection to
     N1-P, repeated here as a dose-response.
   - With the recombined corpus, every arm pilots the same 658 sets at the pool's per-occurrence frequency (X62
     §2: a set is drawn as often as the pool repeats it). The species profile is pool-matched.
2. **The procedural corpus contaminates the meter in proportion to N.**
   - 62 % of H's full sets and 56.5 % of P72's sit verbatim in `n1_proc_v1`, against 10 % and 8 % in the pool (X61
     REVIEW B1, `coverage.log`).
   - So under N1's tiers, U's H and P72 thirds would gain "learned the generator" credit growing with N, exactly
     along the slope X64 reads.
   - The recombined corpus adds no set to any panel's coverage (X62 §2), so U is as clean in A-12.7k as in A-719.
3. **It serves the lever we would fund.** X62 is the recombination lever. A recombined curve says whether its
   corpus size (12,000) sits past the toy's threshold, and whether ≈ 3,000 would do.

**What recombination costs this design.**
- **The corpus is not built.** X64 waits on X62's builder, plus its K3 prefix (§7.1).
- **It cannot touch set novelty.** H and P72 carry 62–69 % sets outside the pool (X62 §1.2). Recombined breadth can
  move them only through a better team-level skill. So G's closable part is mostly F's (70 % pool sets, 26 %
  recombined-like) and the home72 side. §4.3 reads the panels separately.
- **Recombined sixes are less coherent than human-built ones.** The ≤ 2 rule breaks the pool's cores (309 set pairs
  co-occur in ≥ 5 pool teams). The network may learn "recombined-team play". rc72b against F separates the two
  (§4.3).

**The procedural curve is not designed here.** If rc72b reads KILLED (the premium is set-level), a recombined
curve has nothing to close and is not run (§7.1). A set-and-context curve on `n1_proc_v1`'s tiers would then be a
different question, carrying the meter hazard above. It is named, not designed (FINDING 11).

### 2.2 Nested arms with equal per-team draw rates

Every arm trains on the pool, and the larger arms add a prefix of the corpus:
- A-719 = the pool;
- A-3.7k = the pool + K3 (3,000);
- A-12.7k = the pool + ALL (12,000).

**The share is s = K / (K + 719):** 3,000 / 3,719 = **0.8067** and 12,000 / 12,719 = **0.9435**. Under
`gen3_team_mix_v1` ([`../training/team_curriculum.md`](../training/team_curriculum.md), "The team mix"), a fresh
draw takes the corpus with probability s, else the side's usual draw. So every team of the union is drawn equally
often:
- **exactly on the opponent side** (the pool branch is uniform);
- **on the trainee side, to within the pool's 10 % sample bias:** a pool team is drawn 0.9× as often as a corpus
  team, and the sample-bias teams take (1 − s) × 0.1 of the draws (10 %, 1.9 % and 0.57 % by arm; FINDING 9).

**Why nested, not three pure corpora.**
- **The alternative.** Each arm draws only from a corpus prefix of size N, with its own trained panel sampled from
  that prefix: CoinRun's train-levels and test-levels design.
- **Why rejected.**
  - Its smallest arm is not the system we have: every toy and P\* trains on the pool.
  - Its "trained" panel would differ by arm, so G would compare three different panels.
- **What nesting keeps.**
  - home72 stays a trained panel in every arm.
  - A-719 IS the fresh toy C0, reusable as the same-pin fresh control for any later toy arm (X61 Option T's `toy_c0f`).
- **What nesting costs: count is confounded with mix.** A-12.7k's draws are 94 % recombined, A-719's 0 %. The arms
  differ in the human-built share of their teams as well as in N. This is the design's main confound (§8, item 2).

**Why uniform rates and not X62's share 0.5.** At 0.5 a pool team gets ≈ 16× a corpus team's draws in A-12.7k: a
mixture, not N equally weighted contexts. CoinRun and Procgen sample levels uniformly. X62's 0.5 answers a different
question (a plateau fork that keeps the pool's premium).

### 2.3 The argv, the pin, the seeds

- **The toy recipe** is X61's `toy_c0f` derivation (`main.x61.arms`): `rb_toy_ent_s4002`'s `original_command` with
  its run tokens re-typed and the 17 flags the 2026-10-15 deletions removed dropped (inert under
  `--no-damage-op`), `--behaviour-check warn` under X34's hand-read ceilings, `--steps 100000000`,
  `--checkpoint-every-steps 1000000`, `--restart-interval-hours 6`.
- **A-3.7k and A-12.7k** = A-719's argv token for token, plus the `--team-mix` / `--team-mix-sides both` pair of §0.
  A kit test of the `arms_test` kind pins that the three argvs differ by the lever alone.
- **The pin:** one new `main` commit that carries `x62_recomb_v1` (with K3 and its soak) and the arms module, tagged
  on origin and passed to the launcher as `--pin-commit`. It must postdate the 10-26 GIGO fixes (`d8d674b99`; X61
  REVIEW m9). It carries the pinned data root, so `data/` changes on `main` do not reach the runs.
- **Seeds:** 6401–6403 for A-719 and A-12.7k, 6401–6402 for A-3.7k. There is no pairing: a different first team
  draw diverges the runs at once (X61 REVIEW m2). Runs are named `rb_x64t_n719_s<seed>`, `rb_x64t_n3k_s<seed>` and
  `rb_x64t_n12k_s<seed>`; none exists today.
- **The order** interleaves the arms, one run at a time: n719 s6401, n12k s6401, n3k s6401, n12k s6402, n719 s6402,
  n3k s6402, n719 s6403, n12k s6403. A stop at any point leaves a readable subset in which the endpoints are within
  one run of each other.

---

## 3. Dose, seeds, GPU-h

### 3.1 Is 100M enough for the toy to plateau? No

- **The toy's own record.** X34's Tier 1 read GAIN at every check from 210M to 270M, and FLAT only at 280M and 290M
  (ledger 2026-10-20, *X34 TOY: SAME HALVING AND STALENESS*). The toy on the pool climbed for 60M after its step
  size stopped falling. At 100M it was on the rising part.
- **Breadth plateaus later.** Procgen's many-level agents train more slowly early on and rise later (its "implicit
  curriculum"). So at 100M, A-12.7k is likely further below its own plateau than A-719 is (X61 REVIEW m5).
- **What 100M therefore measures.** The gap at a matched budget of 10⁸ steps. That is the field's own regime:
  CoinRun ran 256M, Procgen 200M, at budgets, not at plateaus.
- **The memorisation regime is reached by 100M.** A-719 pilots each pool team ≈ 3,950 times per side, about
  X52's 4,000-pilot practice dose. A-3.7k pilots each team ≈ 760 times and A-12.7k ≈ 220 (from X62 §2's ≈ 395 per
  pool team per 20M at share 0.5). T\* at 290M had ≈ 11,500.
- **The bias, both ways (MY JUDGEMENT).**
  - A-719 has memorised less at 100M than at 290M, so its gap is smaller than T\*'s. That compresses the curve
    toward FLAT.
  - A-12.7k is further from its plateau. That can understate its U, which also biases toward FLAT, or understate
    its home72 loss, which biases away from SHRINKS-HOME.
  - So a FLAT at 100M is weaker evidence than a SHRINKS. The decision rule says so (§4.4, §6).
- **Why not 290M.** About 145 GPU-h at the recommended seeds: the scale of X53-N1, not a screen's.
- **The dose guard.** The trajectory secondary (§4.5) reports whether A-12.7k is still climbing faster than A-719
  at 100M. If a FLAT comes with that flag, a 200M extension of the endpoints is the named follow-up, as a NEW
  registration (≈ +6 × 6.3 ≈ 38 GPU-h).

### 3.2 Seeds

- **One seed per arm (the survey's ≈ 19 GPU-h) is rejected.**
  - A three-point slope with one run per point has one residual degree of freedom, and curvature and seed noise
    cannot be told apart.
  - X61's review found two seeds per arm the minimum that can estimate σ at all (M1, M2). The spread of fresh runs
    is unmeasured; that of forks is not a proxy, because forks share a parent.
- **The endpoints carry the slope.** With x = 2.857 / 3.570 / 4.104, the middle arm adds almost nothing to the
  slope's precision (§5: 3 / 2 / 3 and 3 / 3 / 3 have the same power). It serves the SHAPE (§4.4).
- **Recommended: 3 / 2 / 3** (8 runs). **The budget variant: 2 / 1 / 2** (5 runs). Its slope power is that of
  2 / 2 / 2, and its shape read rests on one A-3.7k seed.

### 3.3 GPU-h (ESTIMATE)

| item | basis | GPU-h |
|---|---|---|
| one fresh toy run to 100M | 100M at the toy C0 forks' 4,607 steps/s ≈ 6.0 h, + ≈ 0.3 h start-up and pool seed (N4 §3.3). The fresh `rb_toy_ent_s4002` ran 2M → 100M in 5.57 h at ≈ 4,880 steps/s (X61 §9). The rate at HEAD is unmeasured (FINDING 4). The 12,000-team table cost N1-P no throughput (1,433 against C0's 1,384–1,424 steps/s; team phase 18.5 s) | **≈ 6.3** (5.9–6.5) |
| **recommended, 3 / 2 / 3** | 8 runs | **≈ 50** (47–52) |
| budget, 2 / 1 / 2 | 5 runs | ≈ 32 (30–33) |
| the survey's 1 / 1 / 1 | 3 runs; not readable (§3.2) | ≈ 19 |
| a 200M endpoint extension (the named follow-up, §3.1) | 6 runs × +100M | ≈ +38 |

- **Disk.** X61's dry-run asked for 20.35 GiB per 100M toy run before thinning. Keep every 25M node to 100M, plus
  the 50M, 75M and 100M nodes the reads use. The planner thins the rest after the read (FINDING 8).

---

## 4. The read

### 4.1 The instruments

- **Kakuna cells.** `main.x35.kakuna` plan / run, the regime of X51 and N4: `metamon:Kakuna` checkpoint 34, greedy
  against greedy, our side greedy on CPU, 20-game role-balanced units, each panel's committed config, rule 5b's
  cores, ≤ 45-min chunks under `mem_cap.sh`.
  - Every X64 cell, K0's included, is played on ONE engine in ONE regime: the regime after the 2026-10-27 Kakuna pack
    change (STATUS §1 note). No banked row from another regime or engine is imported.
- **Counting.** A unit counts iff `main.x35.kakuna`'s rule passes. A timeout is never an outcome. If more than 25 %
  of a final's planned units are uncounted, that final is INCONCLUSIVE.
- **Per final:**

  | panel | units | games | role |
  |---|---:|---:|---|
  | home72 | 60 | 1,200 | G's trained side |
  | H, F, P72 | 40 each | 800 each | U |
  | rc72b | 20 | 400 | G_ctx (reported) |
  | **total** | 200 | **4,000** | |

- **In-house** (`main.h2h play-many`, mirrored pairs; ≥ 500 pairs per cell):
  - **S** on home72 against the six members of the frozen toy R;
  - **S_U**, the same cells on F, H and P72;
  - **G_S** = S − S_U.

### 4.2 The primary statistic

Per final f (arm a, seed s):
- p_{f,k} = wins / counted games (ties are non-wins), for k ∈ {home72, H, F, P72, rc72b};
- U_f = (p_H + p_F + p_P72) / 3;
- **G_f = p_home72 − U_f**;
- its binomial variance v_f = p_h q_h / n_h + (1/9) Σ_{k∈U} p_k q_k / n_k.

**The fit.** Ordinary least squares of G_f on x_f = log10 N_a over all counted finals gives the slope β̂ and the
residual variance s² (n − 2 degrees of freedom). Then:
- σ̂_g² = max(s² − mean v_f, 0), the seed variance of G beyond the binomial;
- σ_g² = max(σ̂_g², (2.0 pp)²): a declared floor, because 6–8 finals cannot rule a smaller spread in (MY JUDGEMENT:
  G is a within-final difference, so it cancels part of a run's overall strength and its σ sits below the 3 pp
  floor X61's review set for a fresh run's level);
- SE(β̂) = √((mean v_f + σ_g²) / Sxx).

**The estimand** is the fitted change over the whole range, **Δ = β̂ × R**, where R = log10(12,719 / 719) = 1.2477.
Its interval is the two-sided 90 % interval Δ ± 1.6449 × SE(β̂) × R (one-sided 95 % on each side).

**Why a log-linear slope and not an endpoint contrast.** With these doses the two have the same power (§5). The
slope uses every final and is the form the literature's curves take (gap against log levels).

### 4.3 The decomposition and the mechanism lines (REPORTED; only §4.4's routing reads one of them)

- **Δ_home** and **Δ_U**: the same fit on p_home72 and on U_f, with the floor 3.0 pp for a level (X61 REVIEW M2).
  Δ = Δ_home − Δ_U, exactly, by linearity.
- **Per panel:** Δ_F, Δ_H and Δ_P72.
  - Context breadth predicts F moves most: 70 % pool sets, 26 % recombined-like.
  - H and P72 move least: their sets are mostly not in any arm's training.
  - A shrink carried by H or P72 alone is a hazard to investigate, not support.
- **G_ctx = home72 − rc72b**, and its slope Δ_ctx. rc72b is a held-out draw of the corpus's own generator (X62 §2
  refuses its sixes), so this is Procgen's test-level gap.
  - Δ_ctx < 0 with Δ ≈ 0 reads as "context generalisation works, but U's deficit is set novelty". That points at the
    set levers (D3, N1-P's sets), not at more recombination.
  - rc72b improving without F improving reads as "learned recombined-team play" (X62's M1 logic).
- **The shape:** φ̂ = (Ḡ_3.7k − Ḡ_719) / (Ḡ_12.7k − Ḡ_719). Log-linear predicts 0.57.
  - φ̂ ≥ 0.8: most of the closure happens by ≈ 3.7k teams.
  - φ̂ ≤ 0.3: the curve is still steep near 12k.
  - Reported only on SHRINKS. It is noisy at two A-3.7k seeds (§5).
- **In-house:**
  - **Δ_S**, the in-pool cost (VGC-Bench's per-team price);
  - **the G_S slope**, the in-house gap. R trained on the pool, so G_S is blind to the gap's level (X37b: P\*'s
    in-house shortfall +4.0 [−0.7, +8.4] against Kakuna's ≈ 11 pp). But A-12.7k's familiarity differs from R's, so
    the slope still reads a CHANGE. It is reported beside Kakuna's, never instead of it.

### 4.4 The classes (one look)

The classes are tested in this order on the 90 % interval of Δ:

| class | rule | reads as |
|---|---|---|
| **SHRINKS** | UB < 0 | the gap closes as contexts grow |
| → **SHRINKS-U** | SHRINKS and Δ̂_U > 0 | it closes from the unfamiliar side: breadth buys U |
| → **SHRINKS-HOME** | SHRINKS and Δ̂_U ≤ 0 | it closes by losing the pool premium: no evidence breadth buys U |
| **GROWS** | LB > 0 | unexpected; check the dose flag (§4.5) and the VOID guards first |
| **FLAT** | LB > −2.0 pp and Ĝ_719's own 90 % LB > 0 | a real toy gap that 10⁴ recombined contexts close by less than 2 pp, at 100M |
| **UNINFORMATIVE** | LB > −2.0 pp and Ĝ_719's LB ≤ 0 | the toy at 100M has no detected gap to close |
| **UNDECIDED** | otherwise | nothing licensed |

- **Ĝ_719's interval** is the mean of A-719's finals' G ± 1.6449 × √((mean v_f + σ_g²) / m_719).
- **The SHRINKS-U / -HOME routing is on a point sign, by design.** Δ_U's 90 % half-width is ≈ 3.0–4.2 pp at 3 / 2 / 3
  (§5), so no interval on it can decide at this size. The routing is registered as a sign and reported with Δ_U's
  interval, and §6 weighs it as such.
- **The −2.0 pp FLAT bar** is the smallest closure that matters against the toy's expected gap (≈ +4 pp at 100M,
  §8): half of it. It is about a fifth of production's ≈ 11 pp gap.
- **Rule 8.** A bound within 1e-9 of 0 or of −2.0, or Δ̂_U within 1e-9 of 0, is not past it: that comparison reads
  UNDECIDED, and SHRINKS routes to SHRINKS-HOME. A final with fewer than 90 % of its planned games, or an arm with
  zero variance, makes the read INCONCLUSIVE.

### 4.5 The trajectory and the dose flag (REPORTED)

- **In-house S and S_U** of every run at 50M, 75M and 100M, against two members of R. These are cheap toy-against-toy
  cells.
- **DOSE-LIMITED** is flagged when A-12.7k's S_U rises from 75M to 100M by more than A-719's plus 1.0 pp (seed
  means). The flag qualifies a FLAT or a GROWS ("at 100M; the broad arm is still climbing") and names the 200M
  extension (§3.1). It never changes a class.
- **The training logs** (bot evals, self-play pool win rates), both arms, for the record.

### 4.6 The K0 gate (CPU, before any GPU)

**What K0 plays.** The three old toy seeds at 100M, read-only copies at pin `3899277a`, X61's K0 set:
- `rb_toy_ent_s4001b` @100,000,169;
- `rb_toy_ent_s4002` @100,000,118;
- `rb_toy_ent_s4003` @100,000,043.

Each plays home72 30 units, rc72b 30 units, and H, F and P72 20 units each: 2,400 games a seed, 7,200 in all.

**K0 answers what X61 never measured:** does the toy at 100M have a gap, and is part of it context?
- **K0 PASS** iff the pooled Ĝ ≥ +2.0 pp AND the pooled Ĝ_ctx ≥ +1.0 pp.
- These are planning bars on point estimates. The pooled SEs are ≈ 1.4 pp (Ĝ) and ≈ 1.7 pp (Ĝ_ctx).
- **On HOLD**, X64 is not launched as registered:
  - a toy whose 100M gap is under 2 pp cannot show a 2 pp closure;
  - a gap with no context part can close only through home72 falling (SHRINKS-HOME by construction).
- **What else K0 gives.**
  - **σ̂_g from the three seeds** (2 degrees of freedom). It is descriptive and never replaces §4.2's floor.
  - **The toy's own H-CTX ratio** at 100M: G_ctx / (home72 − F), on the toy. It is reported beside rc72b's
    production read, and it is never a substitute for it.
- **The caveat:** K0's toys trained at another pin, before the bf16 trunk (X61 FINDING 8). K0 sizes the level; it
  does not stand in for A-719.

### 4.7 VOID guards (a VOID run is re-run, never read)

X61 §8's guards, adapted:
- **V-PIN:** every run's pin is X64's pin.
- **V-ARGV:** `original_command` equals the kit's argv exactly (`main.lineage`).
- **V-MIX:**
  - A-3.7k's log prints the TEAM-MIX line naming `x62_recomb_v1@K3` (3,000 teams, the tier sha) at share 0.8067 on
    both sides, and A-12.7k's names `@ALL` (12,000 teams) at 0.9435, each with 0 pool copies;
  - A-719 has no TEAM-MIX line.
- **V-STEPS:** every final is at ≥ 100,000,000 steps; crash restarts are allowed and counted.
- **V-FAULT:** a core error naming a corpus team voids the run.
- **V-K9:** a K9(b) ceiling crossing stops the run.
- **V-KAKUNA:** every cell shares one regime stamp and one engine.

---

## 5. Power

A Monte Carlo of §4.2–§4.4's rule: 40,000 draws a cell; numpy only; scratch, not committed (FINDING 5).

**The model.**
- The true G is log-linear in N, with A-719's G = +5 pp.
- Half of any change is carried by home72 falling, half by U rising.
- The binomial variance is at §4.1's games, home72 near 0.48.
- Seed noise on G is normal with SD σ_g.

**The rule as simulated:** the floor at 2.0 pp, z = 1.6449, FLAT's bar −2.0 pp. The table gives P(SHRINKS), with
P(FLAT) in brackets.

| design | true σ_g | 90 % half-width of Δ | Δ = 0 | −2 pp | −3 pp | −4 pp | −6 pp |
|---|---|---|---|---|---|---|---|
| **3 / 2 / 3** (≈ 50 GPU-h) | 1 pp | 3.6 pp | 0.015 (0.16) | 0.17 | 0.36 | 0.59 | 0.92 |
| | 2 pp | 3.6 pp | 0.040 (0.20) | 0.20 | 0.34 | 0.52 | 0.82 |
| | 3 pp | 4.7 pp | 0.062 (0.19) | 0.19 | 0.29 | 0.42 | 0.67 |
| **2 / 1 / 2** (≈ 32 GPU-h) | 1 pp | 4.4 pp | 0.014 (0.12) | 0.11 | 0.23 | 0.41 | 0.77 |
| | 2 pp | 4.4 pp | 0.039 (0.15) | 0.15 | 0.25 | 0.37 | 0.66 |
| | 3 pp | 5.7 pp | 0.062 (0.16) | 0.17 | 0.24 | 0.33 | 0.54 |

**What the table says.**
- **X64 detects a large closure and little else.** At the expected −3 pp (§8) the recommended design reads SHRINKS
  about a third of the time. At −6 pp, the size of the whole toy gap closing, it reads SHRINKS 0.67–0.92.
- **A false SHRINKS stays ≤ 0.062.** It exceeds the nominal 0.05 only when the true σ_g is above the floor.
- **FLAT is hard to reach.** Under a true null it reads 0.16–0.20, because the half-width (3.6–4.7 pp) is about twice
  the bar. A FLAT is therefore strong when it comes, and it rarely comes. **Most nulls end UNDECIDED** (≈ 0.75–0.83).
- **Other designs simulated.**
  - At a 1.5 pp floor, 3 / 3 / 3 matches 3 / 2 / 3 to within 0.01 at every cell: the ninth run buys only shape.
  - At the same 1.5 pp floor, 4 / 2 / 4 (≈ 63 GPU-h) reads 0.51–0.80 at −4 pp.
  - Doubling every panel's games shrinks the half-width by only ≈ 12 % at σ_g = 2, because seeds dominate.
- **Δ_U's own 90 % half-width** at 3 / 2 / 3 is ≈ 3.0 pp at a level σ of 2 pp and ≈ 4.2 pp at 3 pp. That is why the
  routing in §4.4 is a sign.

---

## 6. What each class implies for production

### 6.1 What transfers

- **The direction transfers better than the size.**
  - The toy has no belief stack. Production's belief deltas memorise the pool too (FACING;
    [`../model/generalisation_review_2026-10-26.md`](../model/generalisation_review_2026-10-26.md)), a channel X64
    cannot see.
  - The toy's gap is about half P\*'s (X51).
  - Toy-to-production transfer of any lever is uncalibrated (X35's toy calibration NOT SHOWN).
- **The comparison is internal, which helps.** X64's classes compare the toy with itself across N, so a SHRINKS is a
  statement about how this network family responds to context count. It is not a toy lever's effect size carried
  to production. That makes a SHRINKS a fair support for production breadth (MY JUDGEMENT).
- **A FLAT is weaker.** It says "not on the piloting channel, at 100M, at the toy's premium". Production adds the
  belief channel and a premium twice as large.

### 6.2 The table

| X64 class | production breadth (X62, N1-P) | from-step-0 / long-dose breadth (X53-N1; survey rec 2) | corpus size |
|---|---|---|---|
| **SHRINKS-U** | worth production GPU. A plateau-fork NULL from either is read as dose (20M gives ≈ 24 pilots a team; plasticity is NO-LOSS), not as "breadth fails" | **supported:** the long union continuation (rec 2) or X53-N1 is licensed to ask for its GPU, by its own registration | φ̂ ≥ 0.8 ⇒ ≈ 3k teams already suffice; X62's 12,000 is ample. φ̂ ≤ 0.3 ⇒ still steep at 12k; a larger static build (X62's builder takes N) is the corpus question |
| **SHRINKS-HOME** | not licensed by X64. X62's and N1-P's own reads stand on their own rules | not supported: the gap closed by forgetting the pool | — |
| **FLAT** (with Ĝ_719 detected) | X64 adds no support. X62's fork still runs on its own gate tree, because its read is on production F, not on this toy | **not supported:** "breadth from step 0 at 10⁴" loses its toy evidence; X53-N1 stays behind its own gates, with this record against it | — |
| FLAT + DOSE-LIMITED | as FLAT | open: the 200M endpoint extension (≈ 38 GPU-h) is the named follow-up, by a new registration | — |
| **GROWS** | nothing licensed. Investigate: the dose flag, the coherence of recombined teams (rc72b against F), the VOID guards | — | — |
| **UNINFORMATIVE / UNDECIDED** | nothing licensed. The interval goes to the orchestrator; a further seed at the endpoints is a NEW registration, never an amendment after this look | — | — |
| **K0 HOLD** | X64 is not run as registered. The toy at 100M cannot show the curve; the question goes back to production reads | — | — |

**Under each N1-P outcome** (N1-P look 1 is being read, STATUS §2):
- **N1-P GAIN:** X53-N1 is funded on N1-P's evidence. X64 then answers its corpus question: whether a recombined
  corpus closes the toy's gap, and at what count.
- **N1-P NULL:** X64's SHRINKS-U is the evidence that breadth works at a dose the fork did not give. That is the case
  for the survey's recommendation 2 (P\* + ≥ 100M on the union corpus).

### 6.3 What a SHRINKS does NOT license

- **A recipe change.** Nothing joins the production recipe on one read.
- **A goal-(b) claim.** That is Kakuna on production U.
- **Dropping the in-pool guard.** The toy's Δ_home and Δ_S are the expected price (survey §6.1). Production's
  breadth arms keep their non-inferiority co-primary.

---

## 7. Cost, gates and order

### 7.1 The gates, in order (a failure is a HOLD, never a launch)

1. **rc72b (H-CTX, Amendment 2) does not read KILLED.**
   - KILLED means the premium is set-level, and a recombined curve has nothing to close.
   - INCONCLUSIVE or VOID ⇒ hold, as X62 does.
   - This is the gate on which X62 builds its corpus, so X64 adds no corpus build of its own.
2. **`x62_recomb_v1` is built with a K3 prefix tier** and soaked, as X62 §2 specifies.
   X62's revision already refuses rc72b's sixes and applies the STRICT Hidden Power rule (FINDING 3).
3. **The K0 gate passes** (§4.6; CPU, ≈ 19–27 lane-h).
4. **The build:**
   - an arms module modeled on `main.x61.arms`, with three arms, the derivation test and `check --tree`;
   - the pin and its tag;
   - `main.checkargs` exit 0 and `main.launcher --dry-run` "would launch" for each arm at the pin;
   - the `--debug --steps 10000` smoke with the A-12.7k lever under `mem_cap.sh` (X61 FINDING 10 was this smoke
     never run);
   - a first-hour throughput guard: HOLD below 0.7 × 4,607 steps/s.
5. **A PREREG** (§9's skeleton made whole, with the power script committed), then an adversarial review, then the
   orchestrator's GPU GO.

### 7.2 Cost (ESTIMATE)

| item | GPU-h | Kakuna lane-h (266–382 toy games an hour) | other |
|---|---:|---:|---|
| corpus build + soak | 0 | 0 | shared with X62 (≈ 1 agent-day, there); + the K3 tier |
| arms module, pin, transcripts, smoke, PREREG, review | 0 | 0 | ≈ 1.5 agent-days |
| K0 (7,200 games) | 0 | **≈ 19–27** | — |
| training, 3 / 2 / 3 | **≈ 50** | — | ≈ 2.1 card-days, one run at a time |
| Kakuna read, 3 / 2 / 3 (8 × 4,000 = 32,000 games) | 0 | **≈ 84–120** | — |
| in-house S, S_U and the trajectory | 0 | — | ≈ 8–15 lane-h of the S lane (X47: 63 cells in ≈ 50 min on 3 lanes) |
| **total, recommended** | **≈ 50** | **≈ 103–147** | ≈ 6.4–8.2 resource-days |
| **total, budget 2 / 1 / 2** | **≈ 32** | **≈ 71–102** (20,000 + K0) | ≈ 4.3–5.6 resource-days |

**The toy's lane rate is extrapolated.** X51 measured T\* at 7.1 s a game and P\* at 11.8 (1.66×). D3 look 1 measured
production at 160–230 games a lane-h under contention, so the toy is taken at 1.66× that: 266–382 (FINDING 7).

### 7.3 Is X64 a cheaper screen to run before X62's production fork?

| | GPU-h | lane-h | what it answers |
|---|---:|---:|---|
| **X64, 3 / 2 / 3** | ≈ 50 | ≈ 103–147 | the context-count dose curve from step 0, on the toy |
| X62 (5 seeds; X62 §7) | 20.5 | 88–127 (D1's rule) / 126–269 (PR rule) | does recombination move the production PLATEAU on F |
| N4 (the other toy screen) | ≈ 30 | ≈ 26–38 | capacity at the toy plateau |
| the survey's rec 2: P\* + 100M on the union corpus | ≈ 20 per seed (100M at ≈ 1,400 steps/s) | its own read | breadth at production dose |
| X53-N1 (fresh production pair) | ≈ 150–250 to its plateau | ≈ 130–185 | breadth from step 0 at production |

- **Not cheaper than X62.** X62 is a 20M fork, and X64 costs ≈ 2.4× its GPU and about its lane. X64 does not
  replace X62's read either: X62 reads production F against Kakuna, and X64 reads a toy curve.
- **Cheaper than what it gates.** X64 is the screen for the from-scratch and long-dose breadth runs (≈ 40–250 GPU-h).
  It also answers whether X62's corpus size sits past the toy's threshold.
- **The order (MY JUDGEMENT; the queue is the orchestrator's).**
  - X64's GPU can run in any window once gates 1–4 pass, before, beside or after X62's fork.
  - Its Kakuna read competes with X62's, N1-P's, D3 look 2's and H-CTX's for the one anchor lane. That lane, not the
    card, sets when X64 can be read.
  - If the lane is the bottleneck, train first and read later: the finals keep, and K0 has gated the spend.

---

## 8. Predictions (GUESS, written before any number)

| quantity | point | 80 % interval |
|---|---|---|
| K0: pooled Ĝ of the old toys at 100M | +4 pp | [+1, +7] |
| K0: pooled Ĝ_ctx | +2 pp | [−1, +5] |
| **Δ (primary), 719 → 12.7k** | **−3 pp** | **[−6, 0]** |
| Δ_home / Δ_U | −2.5 / +0.5 | [−6, +1] / [−2.5, +3.5] |
| Δ_F / Δ_H / Δ_P72 | +1.5 / 0 / 0 | [−2, +5] / [−3, +3] / [−3, +3] |
| Δ on rc72b (the generator's held-out sixes) | +3 | [−1, +7] |
| φ̂ (shape) | 0.6 | [0.2, 1.0] |
| Δ_S (in-house, in-pool) | −3 pp | [−6, 0] |

**Class probabilities given K0 PASS:** SHRINKS 0.35 (SHRINKS-U 0.20, SHRINKS-HOME 0.15), FLAT 0.10, GROWS 0.02,
UNINFORMATIVE 0.03, UNDECIDED 0.50. **P(K0 HOLD) ≈ 0.3.**

**The basis.**
- T\*'s home72 − F of 3.5 pp, scaled by production's U / F ratio (11.1 / 6.8 pp), gives ≈ 5.7 pp at 290M; ≈ 4 at
  100M at a third of the dose.
- The literature's curve closes most of a gap between 10³ and 10⁴. But recombination reaches only the context part,
  F's share of U and home72's side.
- VGC-Bench, CoinRun and D1 (−1.6 pp) set the in-pool price.

---

## 9. The registration skeleton (for the PREREG unit)

- Arms, argvs and pin as §2.3, the kit transcripts attached. Gates §7.1. Instruments §4.1. The statistic, the
  classes and rule 8 exactly as §4.2–§4.4, with `main.reads` as the read's home (a rule beside N4's proposed `n4_screen`, `d1`'s
  estimators for the per-panel lines).
- The power script and its log committed.
- VOID guards §4.7. Predictions §8, copied verbatim before any run.
- **One look; no look 2.** A further seed is a new registration.

---

## 10. Adversarial self-check

1. **"The gap shrinks because home72 is starved."** It will, partly: A-12.7k gives each pool team 1/18 of A-719's
   draws, and the in-pool price is expected. That is why G cannot license alone. Δ = Δ_home − Δ_U is reported, and
   SHRINKS routes on Δ_U's sign. The routing is weak (a sign on a ± 3–4 pp quantity), and the doc says so rather
   than claiming a decomposition it cannot power. **G_ctx is the better-identified line** (same sets, the
   generator's own held-out sixes) but is not goal (b)'s quantity. Promoting it to a co-primary is the PREREG
   reviewer's question (FINDING 10).
2. **Count is confounded with the team type.** Nested arms move from 0 % to 94 % recombined teams. If recombined
   sixes are easier or harder to pilot, or teach a different style, the slope carries it. The mechanism lines check
   it: rc72b against F, and Δ_F against Δ_H / Δ_P72. The pure-corpus design would remove it at the cost of a
   different "trained" panel per arm (§2.2). Neither is clean, and the nested one keeps the system we have.
3. **100M is pre-plateau** (§3.1). The bias runs toward FLAT, so FLAT is the weaker class (§6.1) and carries the dose
   flag. A SHRINKS at 100M is, if anything, conservative.
4. **The toy's gap may be too small to see.** The X61 review's B2 applies. K0 is the guard, and its bars are planning
   bars on point estimates; at SEs of ≈ 1.4–1.7 pp, K0 can PASS a toy whose true gap is ≈ 1 pp. A-719's own Ĝ test
   inside FLAT is the second guard.
5. **The power is modest.** At the predicted −3 pp the read is SHRINKS about a third of the time and UNDECIDED most
   of the rest, so the expected information is low per GPU-h. The defensible case for X64 is that one SHRINKS-U
   gates ≈ 40–250 GPU-h of production breadth, and that no other read measures a dose-response at all. If the
   orchestrator wants a near-sure answer, 4 / 2 / 4 (≈ 63 GPU-h) is the next step up, and the curve gains little
   from games.
6. **"Cheaper screen" is only half true** (§7.3). On the GPU X64 costs more than X62's fork, and on the lane about
   the same. Its claim is to be the cheap screen for the from-scratch and long-dose runs, not for X62.
7. **σ_g is unmeasured.** The 2 pp floor is a judgement. If the true σ_g is 3 pp, the false SHRINKS rate is 0.062.
   K0's 2-degree-of-freedom estimate cannot settle it, and the read's own residuals have 5 (or 2 in the budget
   variant).
8. **Kakuna's response to recombined teams** (X62's assumption A-K) enters G_ctx through rc72b, not G. G's panels are
   the goal's own.
9. **Recombined breadth cannot reach H's and P72's sets.** If U's deficit is mostly set novelty, G cannot close much
   by construction, and a FLAT would say "context breadth is not the lever" when the truth is "context is not the
   deficit". rc72b's line separates the two (Δ_ctx < 0, Δ ≈ 0), and the K0 bar on G_ctx screens the worst case.
10. **One toy recipe, one pin.** A curve on T-ENT's recipe does not say what a wider or belief-carrying network would
    do. The N4 screen is the capacity read, and the belief channel is production-only.
11. **The post-hoc risk is low.** X64 is designed from a literature recommendation, before any toy breadth run
    exists. Its predictions are in §8.

---

## 11. FINDINGS (standing rule 7)

1. **The survey's ≈ 19 GPU-h (one seed per arm) is not a readable design.** A three-point curve with one run per point
   cannot separate seed variance from curvature. The X61 review set two seeds per arm as the minimum (M1, M2). The
   recommended 3 / 2 / 3 costs ≈ 50 GPU-h; the budget 2 / 1 / 2 ≈ 32.
2. **100M is not the toy's plateau.** X34's Tier 1 read GAIN at 210–270M and FLAT only at 280M and 290M (ledger
   2026-10-20). X64 is a matched-budget read (§3.1).
3. **X62's revision (2026-10-27) refuses rc72b at radius 3 and applies the STRICT Hidden Power rule** (Amendment 2,
   `905c29f5b`; X62 §2.2). STRICT withdraws 86 of the pool's 658 sets and 6 of its 69 species from the recombined
   corpus. So "N is the only variable" holds only approximately: those sets keep only the pool share's exposure,
   ≈ 19 % of A-719's in A-3.7k and ≈ 6 % in A-12.7k. home72 holds none of them. The PREREG must report their share
   of each panel's slots, or the curve may be read on a set-level change that grows with N.
4. **The toy's throughput at HEAD is unmeasured.** 4,607 steps/s is the toy C0 forks at `4609c155`, and ≈ 4,880 the
   fresh `rb_toy_ent_s4002` at `3899277a`. The bf16 trunk and the FP32_ALT τ solve came later. The first-hour guard
   (§7.1) and the first run's dose record settle it.
5. **The power figures come from a scratch Monte Carlo** (numpy, 40,000 draws a cell, the model and rule in §5), not
   committed by this docs-only unit. The PREREG must commit its own script and log.
6. **The toy's home72 − U has never been measured at any dose.** X51 played T\* on home72 and F only. K0 is its
   first measurement.
7. **The toy's Kakuna lane rate (266–382 games a lane-h) is extrapolated** from X51's per-game times and D3's
   measured production lane rate. It is not measured for the toy under today's contention.
8. **Disk:** ≈ 20 GiB per 100M toy run before thinning (X61's dry-run), × 8 runs. The thinning plan (§3.3) must be in
   the PREREG.
9. **The trainee-side sample-bias teams** take (1 − s) × 0.1 of trainee draws: 10 %, 1.9 % and 0.57 % by arm. So their
   exposure falls with N, and their count was not checked here. They are a minor second difference between arms.
10. **Primary choice.** The brief fixed G = home72 − U as the read. G_ctx (home72 − rc72b) identifies the context
    effect better (§10, item 1), but it is not goal (b)'s meter, and rc72b's own production read is still pending.
    It is registered here as a reported line, and the PREREG reviewer should weigh promoting it to co-primary.
11. **The procedural curve is named, not designed** (§2.1). It answers a different question and carries X61 REVIEW
    B1's meter hazard in proportion to N.
12. **This unit** read the docs cited and ran two scratch power Monte Carlos (numpy / scipy, about 1–2 min of CPU,
    no torch import, no GPU, no game). They ran WITHOUT `nice -n 19` and `taskset` on the agent cores, which rule
    5b asks for while a training run is live. They were brief and light, and this is reported as a hazard, not
    claimed harmless. No file under `data/`, `models/` or `src/` changed.

---

## 12. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **This doc CREATED (DRAFT)** (experiment designer, orchestrator brief; docs only) | X64: three nested toy arms (719 / 3,719 / 12,719 distinct teams), fresh to 100M at one pin, the larger two adding X62's recombined corpus at shares K / (K + 719) on both sides so every team is drawn equally often; 3 / 2 / 3 seeds, ≈ 50 GPU-h; the primary is the log-linear slope of home72 − U against Kakuna, its 90 % interval widened by a 2 pp seed floor; classes SHRINKS (→ -U / -HOME by Δ_U's sign), GROWS, FLAT (bar −2 pp, with A-719's gap detected), UNINFORMATIVE, UNDECIDED; gated on rc72b not KILLED, X62's corpus with a K3 tier, and a CPU K0 gate on the old toys at 100M | the procedural tiers of `n1_proc_v1` (set novelty and the H / P72 meter hazard grow with N); three pure corpora (a different trained panel per arm; the smallest arm is not our system); X62's share 0.5 (a mixture, not N equal contexts); one seed per arm (≈ 19 GPU-h; unreadable); 290M (≈ 145 GPU-h); an endpoint contrast instead of the slope (equal power) | [`../research_state/literature/context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md) §7 rec 5 (`5d554c23`); [`design_x62_team_recombination.md`](design_x62_team_recombination.md) §2; [`../research_state/measurements/x61_n1_breadth_2026-10-25/REVIEW.md`](../research_state/measurements/x61_n1_breadth_2026-10-25/REVIEW.md); [`design_n4_capacity_screen.md`](design_n4_capacity_screen.md) §3.3; X51 (`3010c81ba`); ledger 2026-10-20 (X34's Tier 1) |
| 2026-10-27 | **X62's revision propagated** (X62 revision worker, orchestrator brief; docs only) | the corpus X64 borrows refuses rc72b and applies the STRICT Hidden Power rule, which withdraws 86 of the pool's 658 sets from the recombined part; §2's "658 sets in every arm" and FINDING 3 now say so, and the PREREG must report the withdrawn sets' share per panel. Nothing else in this doc changed | — | [`design_x62_team_recombination.md`](design_x62_team_recombination.md) §2.2, FINDING 8 |
