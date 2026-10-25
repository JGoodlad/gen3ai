# REVIEW B: the human-data lever's EXPERIMENTS, attacked on design and causal inference (2026-10-25)

**Under review:** commit `14ae4203` (DRAFT): [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md)
("the lever doc": the ranking, X59, X60, A-1..A-8) and [`human_data_architecture.md`](../../../endstate/human_data_architecture.md)
("the arch doc": Q-H1..Q-H12). Neither doc changed between `14ae4203` and this review's base (`a211e110`).
**Lens:** experiment design and causal inference. Review A (data / bias / leakage) and Review C (reasoning;
[`REVIEW_C_reasoning.md`](REVIEW_C_reasoning.md)) own their lenses. Where a finding touches theirs, this review gives
only its experimental consequence.
**Method:**
- I read the two docs, the beat-Metamon path (goal (b), D1 look 2, the X26c ruling), `UNDERSTANDING.md` §4.8, this
  folder's card and `results.json`, the rung-1b design (§2–§4, §8.5), the HOM's archive and sidecar, `main.x35.d1`,
  D1 look 2's record and inputs, the panel registry, `ou_random_teams.js`, the Metamon checkout's README, and the
  git history of the bit-identity goldens.
- I recomputed every power number on CPU with [`review_b_power.py`](review_b_power.py) (seeded, about 1 min). It
  re-implements X59's rule and `main.x35.d1`'s classifier. Every simulated figure below is quoted from its output.
- No game was played, no model was trained, and no GPU was used. The docs under review were not edited.

## Verdict: UNSOUND as registered. SOUND-WITH-FIXES once B1 and B2 are adopted, and every fix is CPU-only.

The ARCHITECTURE is sound: off by default, one switch, frozen snapshots, the priors rule kept, and one lever per A/B.
The two PRE-REGISTRATIONS cannot answer the questions they are for:
- **X59's rule fires SUPPORTS more often when A-1 is false (pure forgetting: 0.23) than when it is true (real skill:
  0.17).** CONTRADICTS is unreachable (≤ 1.4 % in every world simulated).
- **X60 cannot read NULL.** Its NULL band (±3 pp) is narrower than its own final half-width (3.0 pp), so P(NULL | no
  effect) = 0.000. At the predicted effect it reads UNDECIDED with probability 0.64 (H-BC) to 0.75 (H-TEAMS). That is
  D1's outcome again, for 16–29 GPU-h.
- **Neither experiment answers the owner's two questions** (is human data better than self-play data; does a
  human-only model beat ours). The human-only comparator is not in the ordered plan at all.

**Recommended FIRST experiment: X59-S, the matched-student contest (§3)**, preceded by the HOM's unread trace twin
(about 1 CPU lane-h).
- Two students of our architecture, trained from the same random initialisation on the SAME filtered human states,
  for the same number of steps. They differ ONLY in the label column: the human's action (S-H) or an action sampled
  from P\*'s policy at T = 1 (S-P).
- Read against Kakuna on U and home72, and head to head in-house.
- It answers "a human-only model vs ours" (S-H vs P\*) and "human labels vs our labels at matched data" (S-H vs S-P).
  It tests A-1 with the forgetting confound and the panel contamination cancelled by design, because both students
  see the same states.
- Cost: about 2 GPU-h with a lease, or about 25–35 CPU lane-h without one, plus about 0.5–1 agent-day of build on
  the existing HOM trainer.
- Before it, at zero build: P\* against Metamon's pinned human-only anchor `metamon:SmallRL` on U (§4).

---

## 1. Findings at a glance

| # | sev | finding (one line) |
|---|---|---|
| B1 | **BLOCKER** | X59's SUPPORTS rewards forgetting more than skill. Its power is 0.12, not 0.25, and CONTRADICTS is unreachable. The HOM is a different treatment from H-BC |
| B2 | **BLOCKER** | X60's NULL is unreachable (P = 0.000), and its GAIN power is 0.25–0.36. The binding variance is CPU-cheap Kakuna games, not seeds |
| M1 | MAJOR | No U panel is clean for a human lever. F has human cores near most of its teams, and P72 is drawn from the same Smogon priors C-GAME fills with. The doc's "clean secondary" is wrong by its own card |
| M2 | MAJOR | X60's controls do not isolate "human". H-TEAMS vs OFF re-measures breadth (D1 is the matched non-human corpus). H-BC vs OFF confounds human labels with an extra CE term on new states |
| M3 | MAJOR | A NULL cannot be told apart from too little or excluded data, too small a coefficient, or plateau plasticity. A-8's falsifier is a non sequitur. There is no positive control or dose curve |
| M4 | MAJOR | The ranking: H-BC's rank 1 rests on a false "only lever" claim, misapplied literature and an absent proxy. A cheap HOM-initialised fork (H-INIT) is missing |
| M5 | MAJOR | The owner's questions are under-served. HB0 is not scheduled, Q-H9 has no matched comparator, and Metamon's own human-only models are free comparators the doc does not use |
| M6 | MAJOR | X59's instrument is unchecked. In C-GAME rows a drawn move or mon is never the label, and the HOM's registered trace twin (rung-1b §8.5) has no read. Kakuna is human-pretrained, so it is not a neutral meter for human-likeness |
| m1 | MINOR | X59 arithmetic: the predictions imply Δ_prem −6.8 pp, not −5. "P\*'s X51 cells are fixed" is false (re-play them) |
| m2 | MINOR | Q-H9 as operationalised is toothless. A plateau's self-play buys about +2 pp / 100M, so any positive X60 "beats self-play" |
| m3 | MINOR | TRADE's exchange rate, and the action on a final UNDECIDED, are unregistered. That is D1's FINDING 11 gap again |
| m4 | MINOR | Goal (a) (Kakuna home72, LB 0.549) has no guard, and a pull toward 1500-level play threatens it |
| m5 | MINOR | The C0 reuse check is necessary but not sufficient. It omits data/, the harness, the reference population and the environment. A shared control's error is inherited by every lever |
| m6 | MINOR | AM-5's √φ̂ widening under-covers when seed variance is real: false GAIN 6.5–8 % at seed SD 2.7 pp |
| m7 | MINOR | Five of the eight falsifiers are not falsifiers (A-2, A-3, A-5, A-7, A-8) |
| m8 | MINOR | The flag does not name the rows: the panel registry, bot rule, C-GAME seed and commit all move them (Review C F3). Pin the rows sha in the registration |
| m9 | MINOR | H-BC's dose: an ESTIMATED ~15 passes over 1.7M rows inside the fork, with no held-out human NLL logged. The 10 % loss-fall VOID is checked only after the GPU is spent |
| m10 | MINOR | Card §E's null gradient is biased toward 0 by misclassification (the fit counts are upper bounds), and §E is a post hoc re-cut of held-out dates already read ≥ 3 times |
| m11 | MINOR | The drift alarm is not adjusted for sample size, and species TVD misses behavioural drift |
| m12 | MINOR | X60 forks P\* while production moves on (X26c to 480M). An ADOPT at P\* speaks for P\* only |

---

## 2. Findings

### B1 (BLOCKER) — X59 cannot test A-1: its SUPPORTS class is triggered by forgetting

**Evidence.**
- **The treatment is not H-BC.**
  - The HOM is pure behaviour cloning from P\*: 3 epochs, 10,346 steps × 512 = 5.30M row-visits, lr 1e-4, no RL term
    and no anchor to P\* (sidecar `~/gen3ai_archive/x50_r1b/hom/model/hom_sidecar.json`).
  - Its validation NLL fell from 1.766 to 1.260 within the first quarter epoch, and to 1.156 by the end.
  - H-BC is a small auxiliary term (≈ 10 % of the policy gradient, cosine-decayed to 0) inside PPO, which keeps
    optimising winning. X59 therefore measures "replace P\*'s policy by a human imitation". It does not measure "add
    a human pull to P\*".
  - A SUPPORTS does not imply H-BC helps, and a CONTRADICTS does not imply it fails: a large BC dose drowns RL skill
    that a small coefficient would keep.
- **The statistic confounds the hypothesis with forgetting.**
  - Δ_prem = (HOM_home72 − HOM_U) − (P\*_home72 − P\*_U) shrinks whenever the HOM loses P\*'s home72 MEMORISATION.
    3 epochs of off-pool BC is exactly that.
  - A smaller premium is equally the signature of "the HOM forgot the pool" and "the HOM gained unfamiliar-team
    skill". Only the second is A-1.
  - The registered prediction (HOM_home72 0.52 vs P\* 0.598, Δ_U −1 pp) is itself a forgetting outcome.
- **Simulated class probabilities** (`review_b_power.py`, 200k reps, X51's P\* cells, 400 games per HOM cell):

| world | P(SUPPORTS) | P(CONTRADICTS) |
|---|---:|---:|
| the doc's own prediction (home72 0.52, Δ_U −1 pp) | **0.118** (the doc says ≈ 0.25) | 0.0005 |
| null: HOM = P\* | 0.020 | 0.009 |
| **pure forgetting, no unfamiliar skill** (home72 0.52, Δ_U 0) | **0.229** | 0.000 |
| **real unfamiliar skill, no forgetting** (home72 0.598, Δ_U +4 pp) | **0.167** | 0.000 |
| skill and forgetting (home72 0.55, Δ_U +3 pp) | 0.461 | 0.000 |
| A-1 false, HOM worse on U (home72 0.52, Δ_U −5 pp) | 0.001 | **0.014** |

- **Why it misfires:**
  - The rule rewards the world where A-1 is FALSE more than the world where A-1 is true.
  - CONTRADICTS needs Δ̂_prem > +6 pp, which no plausible world produces. So the "screen against CONTRADICTS"
    (§5.1) screens nothing.
  - Every non-SUPPORTS outcome routes to H-TEAMS first anyway. So X59 changes the build order only through SUPPORTS,
    which is the class most contaminated by forgetting.
- **Two arithmetic slips** (see m1): the 0.25 ignores the conjunction with "Δ_U's LB > −3 pp" (that needs
  Δ̂_U > +1 pp), and the predictions imply Δ_prem = −6.8 pp.

**Fix.**
1. **Replace X59 with X59-S (§3)** as the A-1 test. It holds the states fixed and varies only the labels, from
   scratch, so neither student has pool memory to forget.
2. **If the HOM read is kept** (it is cheap), demote it to DESCRIPTIVE and change its statistic:
   - make **Δ_U = HOM_U − P\*_U** the primary. The HOM is P\* with human labels on human states, and P\* with its own
     labels on those states is P\* itself, so Δ_U IS the label contrast at the fine-tune level;
   - read it one-sided: a Δ_U LB > 0 supports A-1, and a negative Δ_U does not refute it (general-skill loss);
   - extend P\* to the same n on NEW unit seeds (P\*'s cells are not "fixed": the harness is CPU and deterministic, and
     the X26c read `204204bb` reproduced X51's 60 units exactly, so seeds 0–19 add nothing new);
   - at 1,600 games per cell per model (≈ 18 lane-h for both models), the half-width is 2.0 pp and the power at +3 pp
     is 0.84. At 400 it is 4.0 pp and 0.31.
3. **Optional: a play-time dose curve.** Play π_λ ∝ π_P\*^(1−λ) · π_HOM^λ for λ ∈ {0, 0.25, 0.5, 1} against Kakuna on
   U.
   - It is the play-time analogue of a small human pull (H-BC's dose and H-KL's form) at no training cost.
   - It needs an agent wrapper that mixes two networks' logits (≈ 0.5 agent-day; UNVERIFIED that `main.anchors` can
     host one) and costs 2× per decision.

### B2 (BLOCKER) — X60 cannot conclude NULL, and at its predicted effect it most likely ends UNDECIDED

**Evidence.**
- **The frame's widths.** X60 inherits `main.x35.d1`: 140 games per (seed final, panel), NULL iff the widened 95 %
  interval lies inside [−3, +3]. D1 look 2's final half-width was **3.02 pp** (record `reads/d1_look2.json`:
  [−0.92, +5.11]; φ̂ = 0.77, so no widening). Look 1's is ≈ 3.9 pp.
- **Simulated outcomes** (100k reps, the exact two-look procedure with the sign rule and AM-5):

| truth (seed SD 0) | GAIN | NULL | LOSS | UNDECIDED |
|---|---:|---:|---:|---:|
| Δ = 0 | 0.031 | **0.000** | 0.020 | **0.949** |
| Δ = +2.0 pp (the H-TEAMS prediction) | 0.250 | 0.000 | 0.001 | 0.749 |
| Δ = +2.5 pp (the H-BC prediction) | 0.357 | 0.000 | 0.000 | 0.643 |
| Δ = +4.0 pp | 0.706 | 0.000 | 0.000 | 0.294 |

- **What that means for Q-H11.** X60 can say "helps" or "undecided", never "does not help". Q-H11 ("does it help?")
  cannot get a NO, and the plan's routing (an H-BC variant "only after a GAIN or a clear NULL", §4) cannot be reached
  through NULL. Under the doc's own predictions the modal result is D1's UNDECIDED, for ≈ 16 GPU-h (≈ 29 if C0 is not
  reusable).
- **The binding variance is games, not seeds.** D1's per-seed spread shows no overdispersion: φ̂ = 0.77, df 8, ratios
  0.84 / 0.70. Each final's Kakuna cell is binomial-limited, and Kakuna games are CPU.
- **But seeds are bounded loosely.** φ̂'s one-sided 95 % upper bound is 2.26, which allows a between-seed SD of the U
  mean of up to **2.7 pp**. In that world 5 seeds alone leave a ≈ 3.4 pp half-width, and more games do not help
  (420 games: NULL 0.047 at Δ = 0).

**Fix (CPU-only).**
1. **Raise the games to 420 per (seed final, panel)** (21 units: unit seeds 0–20, so the top-up plays seeds 7–20) on every arm.
   - **Top up the 10 existing finals** (C0 s3501–3505, D1 s3501–3505) to 420: +8,400 games ≈ 23 lane-h.
   - If seeds are negligible (the point estimate): half-width 1.74 pp at look 2; NULL 0.58 at Δ = 0; GAIN 0.77 at
     +2.5 pp.
   - At 700 games: NULL 0.84 and GAIN 0.90.
   - **Side benefits:** the top-up shrinks the shared control's error for every lever (m5); it measures the seed
     variance at 3× the resolution; and it re-reads D1 at a 1.74 pp width. That last read must be registered as a new
     descriptive read, because look 2 was D1's final.
2. **Register a seed rule before look 1:** if the topped-up φ̂'s upper bound implies a seed SD > 1.5 pp, add +3 seeds
   per arm or declare before launch that NULL is unreachable. Never discover it afterwards.
3. **Register the action table for EVERY final class**, including UNDECIDED, and the TRADE exchange rate (m3).

### M1 (MAJOR) — No unfamiliar panel is clean for a human lever; the doc's "clean secondary" is contradicted by its own card

**Evidence.**
- **H** (`heldout72_hpfix`) is drawn from Metamon's replay-derived `hl_05_26`. That is declared (lever doc FINDING 3).
- **F** (`forum72`) is not clean, although X60's registered secondary calls F + P72 "populations no human lever trains
  on" (§5.2):
  - card §D: the median forum72 team has **74** ≥ 1500 human sides fitting ≥ 4 of its sets (CORE 263, NEAR 91);
  - 82 % of its teams have ≥ 10 such sides.
- **P72 is drawn from the SAME distribution the lever's rows are filled with.**
  - P72 is `ou_random_teams.js --seed 3721`. Its inputs are `data/pokemon/gen3_teammate_priors.json`,
    `gen3_move_priors.json`, `gen3_item_priors.json` and `gen3_spread_priors.json` (the file's header).
  - C-GAME fills ≈ 52 % of every human side's move slots, every spread, and the unseen mons from those same Smogon
    priors, through the `agents.gen3_data` facade (rung-1b §2).
  - So H-TEAMS literally trains on Smogon-drawn sets, the generator of P72, and H-BC's states carry them.
- **The exclusion leaves the near region.** It is "≥ 4 mons fitting one panel team", so a 3-mon core of an H or F team
  stays in. The HOM's own build found that every one of the 72 heldout teams lost battles to the rule (HOM README
  FINDING 7), so the near region is everywhere.
- **A-7's falsifier has no power.** "The gain concentrates on teams with more near-copies" gets ≈ 10 games per piloted
  team per arm (140 × 5 / 72). A per-team regression's 80 % MDE is **7.5 pp per SD** of the proximity covariate
  (script).
- **Simpson risk.** D1's gain sat on its own population (forum +4.0, H +2.1, P72 +0.1). A human lever's gain will sit
  on H. The equal-weight pooled U can read GAIN from one contaminated panel.

**Fix.**
1. **Correct the secondary's wording:** F and P72 are less-near, not clean.
2. **Make GAIN require** the pooled LB > 0 AND the point estimates on F and on P72 both > 0. That is a registered
   heterogeneity guard, so one panel cannot carry the pool.
3. **Register a per-team proximity score** (the maximum fit count to any human side in the lever's rows after the
   exclusion; a sensitivity at ≥ 3 fits) and report Δ by proximity tercile. Name it descriptive, given its power.
4. **Stake the clean claim on X54's sealed `forum216b`,** with the same proximity report. Note that it shares F's
   population.
5. **Use the H halves as a positive control (M3)**, since H is the dirtiest panel anyway.

### M2 (MAJOR) — X60's control does not isolate "human" for either lever

**Evidence.**
- **H-TEAMS ON vs OFF re-asks D1's question.** D1 asked "does widening the pool with unseen human-built teams at 0.5
  help?" (+2.10 [−0.92, +5.11]; P −1.61). H-TEAMS differs from D1 only in the corpus's population (ladder cores with
  Smogon fills vs forum pastes). The human-data question (Q-H9: is this source better than the non-human alternative?)
  is **H-TEAMS − D1**. D1's five finals exist in the same frame, at the same seeds and dose.
- **H-BC ON vs OFF confounds three things:**
  - (i) the human LABELS (A-1, the claim);
  - (ii) any supervised CE term on states P\* never visits, a representation and regularisation effect;
  - (iii) +20 % learner compute.
  Q-H9's own definition (§2.1: "at a matched GPU budget, spending a share of the update on human rows moves U more
  than spending it on self-play") requires a comparator that spends that share on non-human labels. No arm does.
- **Matched items, checked:**
  - **pin:** C0 at `f4b25238`, D1 at `440b61c3`. Since `f4b25238` there are no commits to
    `learner_golden_test.py`, `learner_golden.json`, `learner_golden_buffer.npz` or `played_bits_sentinel_test.py`,
    and none to `data/teams/` (git log to `a211e110`), so reuse is plausible today;
  - **seeds and dose:** matched in steps, not in compute;
  - **the team pool:** changed by design under H-TEAMS;
  - **the RNG streams:** not addressed. A human-row draw that consumes the env generator would de-pair the arms'
    trajectories.

**Fix.**
1. **H-TEAMS:** register **H-TEAMS − D1** as the human-data contrast (the same widened Newcombe) beside the ON − OFF
   primary.
2. **H-BC:** register a CONDITIONAL placebo arm **P-BC**, run only if H-BC reads GAIN. It has the same human states,
   coefficient, share and schedule, with the labels replaced by actions sampled from the FROZEN P\* at T = 1 (matched
   label noise). H-BC − P-BC is the effect of human labels; P-BC − OFF is the effect of a CE anchor on unseen states.
3. **Add a null-dose identity test to §5.4:** the ON build with coefficient 0 (or share 0) must reproduce OFF's played
   bits exactly. That proves the lever's plumbing does not move the env RNG.

### M3 (MAJOR) — A NULL (or an UNDECIDED) would not say WHY; the design has no positive control and no dose axis

**Evidence.**
- §5.3 concedes that a NULL "does not rule out a fresh-run use". A-8's falsifier ("X60 NULL with HB0 strong would point
  here") does not follow: HB0's strength says nothing about whether a 20M fork from a plateau can move U.
- Four explanations of a NULL are indistinguishable in X60:
  - (a) the useful rows were excluded (FINDING 2 of the lever doc: the exclusion removes the best-covered region);
  - (b) the coefficient is too small (A-5);
  - (c) plateau plasticity (A-8; X52's practised half moved only +3.2 [−0.2, +6.8] with direct practice);
  - (d) the lever is useless.
- The fork is read only at 20M, and no data-scaling curve exists (card §C: "a data-scaling curve is unmeasured").

**Fix (X52's design, inside X60's ON arm).**
1. **Sacrificial halves.** Split H into two halves stratified by `ladder_weights_v1` (and, for H-BC, by human
   near-copy count). The ON arm's rows INCLUDE the ≥ 4-fit human battles near the practised half H-p and EXCLUDE those
   near H-u (registered as a named exception to the contamination guard). H-p's Δ is the lever's ceiling at this dose
   and plasticity. H-u's Δ is transfer. The primary moves to F + P72 (M1).
   - H-p moves, F + P72 flat: the lever works locally, and a NULL elsewhere is coverage, i.e. (a).
   - H-p flat: dose or plasticity, i.e. (b) or (c). Stop the family at the plateau and send it to the fresh-run or
     warm-start route.
   - The cost is that H loses its "unfamiliar" status for this lineage. FINDING 2 of the beat-Metamon doc already moves
     the goal claim to `forum216b`.
2. **Read the 10M checkpoints** as well as the finals (descriptive, CPU): a two-point learning curve.
3. **Offline data-scaling curve (CPU):** train the HOM recipe on 25 / 50 / 100 % of the rows and record the held-out
   NLL, so that "the corpus is too small" has an imitation-side number before any GPU is spent.

### M4 (MAJOR) — The ranking: H-BC's rank 1 rests on a false uniqueness claim, misapplied literature, and an absent proxy; a cheap lever is missing

**Evidence.**
- **The uniqueness claim is false.** "The only lever whose gradient lands on states of teams we never trained" (§0,
  §4) is untrue: H-TEAMS, D1 and X52 all put RL gradient on teams P\* never trained. What only H-BC adds is human
  LABELS on such states. Card §E's proxy for exactly that content is null, and X59 cannot test it (B1).
- **The literature points the other way.** Both precedents cited for H-BC put human data FIRST and keep it, as an init
  plus a KL, through all of RL:
  - AlphaStar (UNVERIFIED, from memory);
  - Metamon. The local README (line 362) says Kakuna REDUCED the human replays' sampling weight while fine-tuning on
    self-play.
  None adds BC to a converged pure-self-play policy. If the cited evidence supports anything, it supports the
  warm-start family (rank 5), not H-BC.
- **Under the doc's own criterion** (pp per GPU-h, with measured anchors), H-TEAMS has the only measured sibling (D1:
  +2.1 n.d., P −1.6) and costs less. H-BC has none.
- **Missing: H-INIT.** Fork the HOM itself (P\* + human BC, already trained) with plain PPO for 20M, in D1's frame. It
  is Metamon's recipe class (human pretraining, then self-play) at no extra build, and it tests AlphaStar's
  forgetting claim directly. The value head was not trained by the HOM and would refit under PPO; that is a declared
  risk.

**Fix.**
- Re-rank with the stated criterion:
  1. the measurements first (X59-S, the Metamon human-only read);
  2. H-TEAMS with the D1 contrast;
  3. H-BC or H-INIT, whichever X59-S favours (human labels better ⇒ H-BC; human labels worse but the HOM's Δ_U ≥ 0
     ⇒ H-INIT);
  4. H-KL (see m-lit).
- Strike "the only lever".

### M5 (MAJOR) — The owner's two questions are not answered by anything in the ordered plan

**Evidence.**
- §0's order is X59 → toggle → H-BC or H-TEAMS → X60. HB0 (Q-H10) appears only in a cost row (≈ 1 GPU-h + 6 lane-h),
  never in the order, although the owner asked it explicitly and it is the cheapest GPU item in the plan.
- Q-H9 ("better than self-play?") has no matched comparator anywhere (M2), and m2 explains why X60's GPU-h framing
  cannot answer it either.
- **The doc misses free comparators.** Metamon ships human-ONLY models: `SmallIL`/`SmallRL` (15M), `MediumIL`/`MediumRL`
  (50M) and `LargeIL`/`LargeRL` (200M), all "trained on 1M human battles" (README lines 322–326). Two consequences:
  - `metamon:SmallRL` is already a pinned anchor of ours (EXTERNAL_ANCHORS_SOP tier B), so P\* vs SmallRL on U is a
    zero-build read.
  - Metamon's own local GXE table (README "Early Gen OU Local GXE", G3, modern-replays team set) puts its human-only
    models far below its human-plus-self-play ones: SmallILFA 38 %, LargeRL 41 %, against Superkazam 62 % and Kakuna
    67 %. P\* beats Kakuna on home72. That is a strong prior, from Metamon's recipe class at up to 200M, that a
    human-only model does not beat ours. It is README-sourced and transitive across meters, so it is UNVERIFIED for
    our cells.

**Fix.**
1. **Zero build:** P\* vs `metamon:SmallRL` on H, F, P72 and home72, 400 each (≈ 1.3 lane-h at ≈ 3 s a game).
2. **Small build:** pin `LargeRL` and `LargeIL` in `designs/ops/anchors.json` (two downloads; the owner's or the
   orchestrator's call) and read P\* against them on the same cells (≈ 4.5 lane-h each).
3. **Architecture-matched:** X59-S's S-H vs P\* is HB0 in all but name.

### M6 (MAJOR) — X59's instrument is unchecked: a structural label leak, an unread trace twin, and a non-neutral meter

**Evidence.**
- **A structural leak in C-GAME rows (exact, not statistical).** Every own move the log reveals is kept, and the rest
  are drawn. A move the human chose is revealed by being used, so **a drawn move is never the label**. Likewise a mon
  that was switched in is revealed, so **a drawn bench mon is never a switch target**. In 29–32 % of ≥ 1500 sides
  some mons are drawn whole, and ≈ 52 % of move slots are drawn.
  - A model trained on these rows learns "an option that looks Smogon-typical and unused is never chosen".
  - At play time every slot is real, so the bias costs it play in a way that has nothing to do with A-1.
  - Whether the draws are distinguishable is unmeasured. Our observation has no own-side "revealed" bit (searched
    `src/agents/observation/`), so any signature is statistical.
- **The registered check has no read.** The rung-1b design registered a TRACE TWIN (§8.5: the HOM's NLL gap between
  completed and true rows of the same decisions, minus P\*'s). I found no read of it in `x50_r1b_hom_2026-10-23/` or
  anywhere under `measurements/` (UNVERIFIED that it never ran).
- **Kakuna is not a neutral meter for human-likeness.** It was trained on Metamon's parsed human replays (README
  line 825). A human-imitating policy may be more predictable to it than P\* is. That biases X59, HB0, H-BC and H-KL
  together, in an unknown direction.

**Fix.**
1. **Read the trace twin before any HOM-based or C-GAME-trained read** (forwards only, ≈ 1 lane-h). Register a bar: if
   the HOM's twin gap exceeds P\*'s by more than 0.05 nats, treat the HOM's Kakuna score as a lower bound.
2. **In X59-S the leak is MATCHED** (both students see identical completed states), so the contrast survives it, and
   only absolute levels are affected. Say so.
3. **Add one non-human-trained meter to every human-lever read:**
   - in-house head-to-head on U against P\* (fast, high n); and/or
   - Foul Play at a fixed `--search-time-ms` (EXTERNAL_ANCHORS_SOP tier C).

### Minor findings

- **m1 — X59 arithmetic.**
  - The point predictions (HOM_home72 0.52; P\* home72 0.598 and U 0.487; Δ_U −1) give Δ_prem = (0.52 − 0.477) −
    (0.598 − 0.487) = **−6.8 pp**, not −5.
  - "P\*'s X51 cells are fixed, so doubling narrows only to 6.5 pp" is wrong: extending P\* to 800 a cell on new unit seeds gives 5.6 pp
    and P(SUPPORTS) 0.25 under the prediction (script). It is moot after B1.
- **m2 — Q-H9's definition has no teeth.** At a plateau, more self-play buys ≈ +2.05 pp / 100M (the drift read
  `dbc49cc1`), so the matched-compute self-play alternative to H-BC's +20 % learner time is worth ≈ 0.1 pp. Under §2.1,
  any positive X60 makes human rows "better than self-play". **Fix:** define Q-H9 against the matched placebo or
  corpus (P-BC, D1; M2) and the matched students (X59-S).
- **m3 — Unregistered discretion.** TRADE "reports the exchange rate" with no accepting rate, and nothing says what a
  final UNDECIDED triggers. D1 hit exactly this (beat-Metamon FINDING 11; the orchestrator had to rule). **Fix:**
  register both before launch, for example ACCEPT-TRADE iff Δ̂_U ≥ 2 × |Δ̂_P| and the goal (a) guard holds; UNDECIDED
  ⇒ the named next step.
- **m4 — Goal (a) has no guard.** Kakuna home72 is a "registered secondary" with no rule. H-BC's main declared risk is
  a pull toward 1500-level play, and goal (a)'s margin is LB 0.549. P (in-house, half-width ≈ 1.3 pp at D1) is the
  sensitive harm meter, but it does not measure Kakuna. **Fix:** a guard that ON's Kakuna home72 at the final (400
  games pooled over seeds) keeps its Wilson LB > 0.50, reported with its width.
- **m5 — The C0 reuse check (§5.4).**
  - It checks two goldens and the argv. It should also check:
    - `data/` (pool and priors) content hashes;
    - the Kakuna harness commit used to read C0 (and its units / seeds 0–6);
    - the frozen reference population's sha for P;
    - the sim binary hash;
    - torch, CUDA and driver versions.
  - **The shared-control risk.** One control serves D1 and both X60 levers, so its error enters every Δ with the same
    sign. C0 reads 0.471 on U against P\*'s 0.487 at X51 (a different harness and units, n.d.). The top-up in B2 is
    the fix.
- **m6 — AM-5 under-covers.** It multiplies by √φ̂ with no small-df quantile. With a true seed SD of 2.7 pp the
  simulated false-GAIN rate at Δ = 0 is 6.5 % (140 games) and 7.9 % (420 games), against ≈ 3 % with no seed variance.
  **Fix:** scale by an upper-quantile φ (F or χ² at the recorded df) or use t_df.
- **m7 — Falsifiers.**
  - A-2 has no test: the trace twin (M6) is its test.
  - A-3 is a point estimate ("Δ_P < −2 pp"); make it "Δ_P's 97.5 % UB < −2 pp".
  - A-5 reuses A-3's event.
  - A-7 is unpowered (M1).
  - A-8 is a non sequitur (M3).
  Each needs an interval statement and its power.
- **m8 — Rows identity.** `bc:hr_spec_v1@0.05` does not determine the rows: the registered-panel set (the exclusion
  widens with every new panel), the bot rule, the C-GAME seed, the extraction commit and the encoder layout all move
  them. The rows sha is stamped only at launch. This agrees with Review C F3. **Fix:** name the rows manifest sha in
  X60's registration and VOID on mismatch.
- **m9 — H-BC's dose inside the fork.** At production `n_epochs` 5 with a 25 % human minibatch per PPO minibatch,
  20M steps visit ≈ 25M human rows, ≈ 15 passes over 1.7M (ESTIMATE). The HOM's held-out NLL was still falling at 3
  passes, but 15 risks memorising the corpus. **Fix:** log a held-out human NLL (the 2 % validation battles) during
  the fork, and move the "BC loss falls ≥ 10 % in 1M" VOID into the pre-launch smoke (≈ 0.2 GPU-h), so that a VOID
  never costs a 16 GPU-h arm.
- **m10 — Card §E.** A human side's "fits" are UPPER bounds (an unrevealed move fits any set). So the "pool-like ≥ 4"
  stratum contains truly unfamiliar teams whose sets stayed hidden, and that misclassification biases the gradient
  toward 0. The held-out dates 09-21..23 have now been read by R1, R-b1 and §E. **Fix:** re-cut §E on rows whose
  acting mon has all four moves revealed (census tier A, 18 %), and stop calling the held-out days "untouched".
- **m11 — The drift alarm.** 0.074 is fixed, but the split-half floor scales roughly as 1/√n (0.009 at 59k sides;
  ≈ 0.03 at 5k, the archive's monthly size). Species TVD also cannot see set- or behaviour-level drift. **Fix:** an
  alarm relative to each snapshot's own floor, plus a behavioural meter: the HOM's NLL on each new month's ≥ 1500
  rows.
- **m12 — The base checkpoint.** X60 forks P\* (280M) because C0 does. Production moved to X26c (480M, ruling
  2026-10-24). Its read (`204204bb`, landed during this review) is FLAT on U against P\*: +0.08 pp [−3.90, +4.06],
  with S1 not rising. So forking P\* loses little on goal (b)'s axis, and the "≈ 4 pp short" baseline stands. An
  ADOPT at P\* must still be re-read at the production checkpoint before it steers production.
- **m-lit — Decisions that rest on UNVERIFIED literature** (the lever doc §4):

| claim (as cited) | decision resting on it | the risk | the cheapest check |
|---|---|---|---|
| piKL bought human-compatibility, not strength against a fixed agent | H-KL ranked 4th | Kakuna is human-pretrained, so "compatibility" and "strength against this opponent" may not separate. My recollection (UNVERIFIED) is that human-KL-regularised RL (Diplodocus, Bakhtin et al. 2023) was reported strong in human-population play | the λ logit mix (B1 fix 3) is H-KL's play-time analogue, on CPU |
| AlphaStar: RL forgets a supervised init without a KL | the warm start ranked 5th | H-INIT (M4) tests it at 16 GPU-h, not ≥ 42 | H-INIT |
| AlphaStar's KL and Metamon's pretraining "keep a human term" | H-BC ranked 1st | both are init plus KL throughout, not an auxiliary term added at a plateau (M4) | none: drop it as support |
| Metamon reports RL variants above IL | HB0-ORL as the follow-on | partly CHECKED: README's local GXE has LargeRL 41 % vs SmallILFA 38 % (G3), a small gap, and both are far below the self-play-trained models | the Metamon human-only read (M5) |

---

## 3. The recommended FIRST experiment: X59-S, the matched-student contest

**The question.** On the SAME human states, does a policy taught by human labels play unfamiliar teams better than one
taught by our own policy's labels? And does a model trained only on humans beat P\*?

**Design.**
- **Rows.** `hr_spec_v1`'s TRAIN dates; the actor rated ≥ 1300 (the card's HB0 stratum); the card's DROP filters (bots,
  walk-overs, duplicates); the panel exclusion over EVERY registered panel, `forum216b` included; C-GAME-completed; ≥ 2
  legal actions. The rows manifest sha is registered.
- **Two students, ONE difference.**
  - **S-H:** labels = the human's action.
  - **S-P:** labels = one action sampled from FROZEN P\*'s masked policy at T = 1 on the same row, keyed by the row id.
    A sample, not the argmax and not soft targets, so the two label columns carry the same kind of noise (the
    matched-noise rule).
  - Both students: our architecture (`--arch production`'s network); the same random init (one seed); the same batch
    order (one keyed draw); the HOM's optimiser; a FIXED step budget with no early stopping (early stopping on each
    arm's own validation labels would de-match the arms).
  - One epoch on CPU, or three under a GPU lease, for both.
- **The optional third arm S-S:** P\* labels on P\*'s own SELF-PLAY states (pool teams), the same number of rows. S-P −
  S-S is the value of human STATE coverage at a matched teacher, which is the breadth mechanism H-TEAMS and D1 rely
  on.
- **Instrument first:** the HOM's trace twin (M6), forwards only.

**Reads.**
- Each student against Kakuna ckpt 34, greedy against greedy, X51's harness `63f97a57`, seeds 0–19: home72 400, and H,
  F and P72 at 800 each. That is 2 × 2,800 = 5,600 games ≈ 16 lane-h.
- In-house (`main.h2h`, CPU, thousands of games): S-H vs S-P head to head on each U panel and on home72, and each
  student vs P\*.
- S-H and S-P vs `metamon:SmallRL` on U is optional (the human-only anchor; M5).

**Statistics.**
- **Primary:** Δ_S = S-H − S-P on U against Kakuna (equal-weight panels, the Newcombe of `main.x35.d1` with no seed
  widening, since there is one seed per arm). At 2,400 U games per student the 80 % MDE is **4.0 pp**; at 1,200 it is
  5.7 pp (script).
- **Floor fallback:** if both students score < 0.15 against Kakuna on U, the Kakuna contrast is floor-limited, and the
  in-house head-to-head on U decides (registered now).

**Rule** (rule 8: a bound within 1e-9 of 0 is the conservative class):
- **HUMAN-LABELS-BETTER** iff Δ_S's LB > 0. A-1 is supported at first order: build H-BC first, with the P-BC placebo
  conditional (M2).
- **OWN-LABELS-BETTER** iff Δ_S's UB < 0. A pull toward human labels on human states moves P\* toward worse-on-U labels:
  drop H-BC. If the HOM's Δ_U (B1 fix 2) is ≥ 0, H-INIT is the remaining label-route lever.
- **else UNCLEAR:** H-TEAMS first, with the D1 contrast (M2).
- **Q-H10, reported:** S-H − P\* on U and home72 against Kakuna, with intervals. "A human-only model beats ours" iff
  the LB is > 0 on U (not expected, M5).

**Why it is a better test than X59.**
- Neither student has pool memory, so the forgetting confound (B1) is gone.
- The states are identical, so the panel-proximity contamination (M1) and the C-GAME leak (M6) affect both arms
  equally and cancel in Δ_S.
- The contrast is labels only, which is exactly H-BC's distinctive content (M4).
- It answers both of the owner's questions at matched data, and no GPU-h framing is needed.

**What it cannot say.** From-scratch students are not a plateau fork. A HUMAN-LABELS-BETTER result does not show that
H-BC inside PPO helps; X60 (fixed) does that. It is one seed per arm. Per the strong-claims rule, a decisive verdict
(either way) that steers a GPU build is replicated with a second seed pair first (+50 % cost).

**Cost.**
- P\*'s labels on the rows: a forward only, ≈ 2–4 CPU lane-h (ESTIMATE), reusing R-b1's cache where it covers.
- Two students: ≈ 5–10 CPU lane-h each at one epoch on ≈ 6.9M rows (ESTIMATE from the HOM's 373 rows/s; the HOM took
  8.7 h wall for 5.3M rows under contention), or ≈ 1 GPU-h each with a lease.
- Reads: ≈ 16 lane-h.
- Build: ≈ 0.5–1 agent-day (the HOM trainer with a label-column switch and a from-scratch init).
- In total, ≈ 25–35 CPU lane-h, or ≈ 2 GPU-h + 20 lane-h.

**VOID.**
- The label columns differ on rows other than the label: assert the row bytes are equal except the label.
- A student fails to load in the anchors harness.
- A 20-game smoke is not reproduced by a seeded repeat.
- More than 1 uncounted unit per cell.

## 4. The cheapest decisive reads for the owner's two questions, in order

1. **"Does a human-only model beat ours?"**
   - P\* vs `metamon:SmallRL` (human-only offline RL, already pinned) on H, F, P72 and home72: zero build, ≈ 1.3
     lane-h.
   - Then `LargeRL` / `LargeIL` (200M, human-only) once pinned (two downloads).
   - Then X59-S's S-H vs P\* (our architecture).
2. **"Is human data better than self-play data?"**
   - X59-S's Δ_S (labels at matched states) and its optional S-S arm (states at a matched teacher).
   - Then, inside training: X60 with the D1 contrast (H-TEAMS) or the P-BC placebo (H-BC), powered per B2.

## 5. X60, as it should be registered (summary of the fixes)

| item | as drafted | fixed |
|---|---|---|
| games per (seed final, panel) | 140 | **420** (top up C0 and D1 too; ≈ 23 + 17 lane-h) |
| NULL reachable at Δ = 0 | 0.000 | 0.58 (seed SD ≈ 0); a seed rule if φ̂'s UB says otherwise |
| primary | U = H, F, P72 pooled | F + P72 pooled + the heterogeneity guard; H split into H-p (ceiling) and H-u (transfer) |
| human-data contrast | none | H-TEAMS − D1; H-BC − P-BC (conditional) |
| guards | G-E, G-C, G-F | + goal (a) Kakuna home72 LB > 0.50; + the null-dose identity test |
| decision table | GAIN / TRADE / NULL / LOSS / UNDECIDED, nothing adopted by rule | + a registered exchange rate; + the action for each final class |
| learning curve | the final only | + 10M checkpoints (descriptive) |
| VOID timing | after launch | the BC-loss and share checks in the pre-launch smoke |

## 6. On the complexity budget and snapshots (the brief's question)

- **One flag, one id: not quite.** The flag is one token, but the id does not pin the rows (m8). The budget holds once
  the registration names the rows sha.
- **Comparisons across snapshots under drift.** The meters (frozen panels, frozen Kakuna) do not drift, so two
  snapshots' lever results are comparable ONLY as two A/Bs against a common control at a common pin. A verdict on
  `hr_spec_v1` speaks for May–September. At ≈ 0.04 TVD a month (4–5× the floor) a new snapshot is a new treatment,
  never a replicate (the "replicate = same pin + seed" rule). The arch doc's §6 says this correctly. The gaps are the
  floor scaling and the missing behavioural meter (m11).

## 7. Reproduce

```
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES=""
nice -n 19 taskset -c 5-7,13-15 $PY -I designs/research_state/measurements/human_replay_quality_2026-10-25/review_b_power.py
```

Seed 20261025; numpy + scipy; ≈ 1 min on three cores. The D1 inputs are `x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json`
and `reads/d1_look2.json` (φ̂ 0.7704, df 8).

## 8. FINDINGS (standing rule 7)

1. **UNVERIFIED: that the HOM's trace twin was never read.** I searched `measurements/` and the HOM's README and found
   no read, but a read outside the repo cannot be excluded.
2. **UNVERIFIED: `main.anchors` hosting a logit-mixing agent (B1 fix 3) and the `LargeRL` / `LargeIL` downloads (M5).**
   The latter need an `anchors.json` pin and the orchestrator's (or owner's) word on the download.
3. **UNVERIFIED: the literature.** The AlphaStar and Diplodocus readings are from memory. The Metamon facts are from
   the local checkout's README (`~/dev/metamon`, read-only), and its GXE table is Metamon-local, not our meter.
4. **The power model is binomial.** It assumes no per-unit extra variance beyond the seed term. X51's paired
   half-widths (±6.8 pp at n = 400) equal the unpaired binomial ones, which supports this for Kakuna cells. The seed
   SD is bounded only by D1's df-8 φ̂.
5. **The cost figures marked ESTIMATE** (student training throughput, P\*-labelling forwards, H-BC passes) are
   arithmetic from the HOM's logged rates, not measured runs.
6. **Process.** CPU only (`CUDA_VISIBLE_DEVICES=""`, rule 5b's cores, `nice 19`). No game, training, GPU, download or
   launch. `models/`, `data/`, the corpus, the X50 archive and the Metamon checkout were only read. The docs under
   review were not edited.
