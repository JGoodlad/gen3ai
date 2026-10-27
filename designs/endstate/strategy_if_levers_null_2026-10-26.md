# If D3, N1-P and X59-S all come back without a GAIN: the plan for goal (b)

**Status: REVISED DRAFT (2026-10-26), explicit-only. Nothing is built, registered or launched.** It is a contingency
plan for the MODAL outcome of the three reads in flight, revised after the adversarial review
[`strategy_review_2026-10-26/REVIEW.md`](../research_state/measurements/strategy_review_2026-10-26/REVIEW.md)
(`6fca2c06`: B1 + M6–M9 + m9–m12 on this doc; §9 disposes of each). Goal (b), its meter and every read it cites are
[`design_beat_metamon_path.md`](design_beat_metamon_path.md)'s. The levers N1–N5 are the survey's
([`sota_survey_2026-10-26.md`](../research_state/literature/sota_survey_2026-10-26.md) §5). Search is
[`search_ledger.md`](search_ledger.md) and [`design_search_next.md`](design_search_next.md). The from-scratch breadth
pair is [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md). The revision's Monte Carlos are in
[`strategy_revision_2026-10-26/`](../research_state/measurements/strategy_revision_2026-10-26/). The piloting-gap
analysis and the engine-mix audit had NOT landed when this was revised (§8 FINDINGS 2–3).

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = X26's plateau
(`rb_x26_s2601` @280M); C0 = X35's plain-continuation control; U = the three unfamiliar panels pooled (goal (b)'s
meter); F = `forum72`; H = `heldout72_hpfix`; P72 = the procedural panel; S = in-house strength; σ_s = the
between-seed SD of a fork final's true F score; GPU-h = hours of our local card; **lane-h = one wall-hour of the
Kakuna anchor lane at `--concurrency 1` on rule 5b's agent cores, at the MEASURED 160–230 F games an hour** (D3 look 1:
≈ 2,660 games in ≈ 11.7 h, units 9–31 min, 27–93 s a game; N1-P look 1: 486 units ≈ 9,720 games projected at
≈ 60 h, `designs/tools/reads.md` §3; the review's 360 games a lane-h is a quiet-box figure, 1.6–2.2× optimistic);
**resource-day** = (GPU-h + lane-h) / 24, both lanes being single and scarce (MY JUDGEMENT); **GUESS** = an
unmeasured effect; **ESTIMATE** = a cost from measured rates; **MY JUDGEMENT** = an inference no read states.

---

## 0. One screen

- **The scenario is the likeliest one.** The registered priors give N1-P a Kakuna GAIN with 0.20 and D3 with 0.15;
  X59-S's outcome is not a Kakuna GAIN at all, and HUMAN-DATA-BETTER is perhaps 0.15 (GUESS). "All three miss" means
  no Kakuna GAIN from D3 or N1-P and X59-S not HUMAN-DATA-BETTER: **≈ 0.58 if independent, ≈ 0.68 if D3's and N1-P's
  misses are perfectly correlated** (they share frame, control and meter).
- **What "null" would then mean.** Every data-side lever tried at the plateau reads within ≈ ±2 pp, none resolved.
  Five explanations remain (§2): **plasticity**, **capacity**, **resolution** (the levers help by +1–2 pp and a
  ±4 pp read cannot see it), **dose** (20M forks give each new team ≈ 24 pilots), and **the meter**.
- **THE FIRST BRANCH: PR-1, the zero-GPU PRECISION READ** (§3.1). Replay the banked 20M-fork finals of the breadth
  family (D1, D3, N1-P) and the shared C0 on F against Kakuna, at ONE engine stamp, with new unit seeds, and read ONE
  registered estimand: the breadth-family mean contrast. **38,000 F games** (8,000 per breadth arm, 14,000 for C0).
  It detects a family mean of +1.5 pp with **0.76** (σ_s 0) / **0.71** (σ_s 0.5) / **0.57** (σ_s 1); false GAIN
  0.02–0.04; under a true null it reads KILL (UB < +1.5) 0.61–0.79 of the time, half of those at a binding half-way
  futility look. **CPU: 165–238 lane-h (≈ 7–10 days of the anchor lane locally; ≈ 126–181 expected under a null),
  0 GPU-h** — or ≈ 1 day on a rented CPU box for ≈ $5–25 (UNVERIFIED), which needs the owner's approval.
- **The revised top three by P(GAIN) per resource-day** (§3.2), each after its zero-GPU gate:
  1. **PR-1** itself (zero GPU). It is information, not a lever: it decides whether the six breadth rows are worth
     any GPU, and it banks a 14,000-game C0 at the tree's engine stamp that every later fork reuses.
  2. ~~**N2, the training-only set-channel mask**~~: **DROPPED 2026-10-26.** Its registered exact-set identity test
     read COMPOSITIONAL: a familiar set's Smogon re-draw moves P\* only 1.19× [1.07, 1.33] an unfamiliar one's,
     against a 1.5× bar ([`n2_identity_test_2026-10-26/README.md`](../research_state/measurements/n2_identity_test_2026-10-26/README.md)).
  3. **The breadth STACK fork, only on PR-1's BREADTH-GAIN**: P(GAIN) 0.62 at the same cost (≈ 0.12 per
     resource-day). Without PR-1 it falls to 0.49 and is not run.
- **The tree** (§4): T1 is zero-GPU (PR-1, the N3 probe, search's E2; N2's test ran 10-26 and dropped N2; **the
  plasticity probe ran 10-27 and read NO-LOSS**, so T3 takes the capacity branch). **The owner is asked at the
  END OF T1, with an information trigger** (§5), not when a GPU budget runs out: PR-1 KILL with no plasticity LOSS ⇒
  at once (the plasticity half of that trigger now holds); otherwise T2 runs under a cap of ≈ 40 GPU-h and ≈ 250 lane-h without a GAIN.
- **The only bet whose expected value alone reaches the ≈ 4.3 pp gap** is still the scale bet (a wider network from
  scratch on 10⁴ teams to its plateau, ≈ 150–250 GPU-h). It is the owner's call, realistic only with a second card.

---

## 1. What each in-flight read leaves open if it misses

| read | what a miss looks like | what it rules out | what it does NOT rule out |
|---|---|---|---|
| **D3 look 2** (X41, set mutation; look 1 CONTINUE: H +1.27 [−2.63, +5.16], P −0.24 NON-INFERIOR) | UNDECIDED or NULL (its look-2 classes at its look-1 posterior; not recomputed here, FINDING 4) | an exact-set-identification fix worth ≥ ≈ 3 pp at 0.3 share, 20M, at the plateau | a +1–2 pp effect (P costs nothing: a free stack member) |
| **N1-P** (12,000 procedural teams at the plateau; trained, read in flight) | NULL (0.30) or UNDECIDED (0.47) | "N1 at the plateau" at this share and 20M (its PREREG §6) | N1 from step 0 (plasticity), more dose, a human-team corpus |
| **X59-S** (three matched students, training now) | NOT-RESOLVED on Δ_data; UNCLEAR on Δ_labels | nothing alone: its UNCLEAR routes to **H-TEAMS first with the D1 contrast**; only SELF-PLAY-DATA-BETTER, with Δ_labels not HUMAN-LABELS-BETTER and Δ_states' LB ≤ 0, closes human data at this volume | the human prior at a larger volume; offline RL (FR-3: on the same human data offline RL beats imitation by ≈ 16 pp) |

Not re-opened by this plan, with their status stated exactly:
- **The belief channel: NOT DETECTED at ±4–5 pp, not ranked.** λ = 0.25 read U +1.50 [−2.74, +5.74], λ = 1 read
  −1.50 [−5.47, +2.47] (`belief_prior_mix_2026-10-26/`): an effect below ≈ 5 pp is not excluded. It is not ranked
  because X37a puts the gap in PILOTING, not FACING, and no lever on it is cheaper than the rows below.
- **More plain training** (X26c FLAT, +0.08 [−3.90, +4.06]); the archive and SPRT pools (NULL); exploiters and X49
  (they serve facing or exploitability); a different opponent (PastaMittens ≈ Kakuna); training against Kakuna
  (REJECTED: it trains on the meter).
- **"Human data alone beats us".** P\* beats every human-only Metamon model (U 0.631–0.790); the human prior is
  only a candidate as an ADDITION to self-play.

---

## 2. The five explanations, and the cheapest read that separates them

| explanation | predicts | the cheapest separating read | cost |
|---|---|---|---|
| **Resolution** | the breadth levers help by +1–2 pp each; a big enough read sees it | **PR-1** (§3.1): the breadth family on F at 38,000 games | 0 GPU-h, 165–238 lane-h (or ≈ 1 CPU-box day) |
| **Plasticity** | the plateau resists data a young network would absorb | **the plasticity probe**, registered bars (below), on X26 @25M / @100M / P\* (all banked). **READ 2026-10-27: NO-LOSS** ([`plasticity_check_2026-10-26/README.md`](../research_state/measurements/plasticity_check_2026-10-26/README.md)): P\*'s end-to-end refit AUC on its own greedy labels 0.986× @25M's, near-dormant 0.78×; on a neutral target the trunk refits slower than @25M (1.11×) but that was paid by 175M and does not grow to 480M; its behavioural complement is a GPU read (N1-P's argv forked from X26 @100M vs from P\*, 20M each at the X35 pin: the Δ difference is plasticity in the meter's units) | probe ≈ 1 agent-day, ≈ 5–10 CPU-h; complement ≈ 8–16 GPU-h + ≈ 60 lane-h |
| **Dose** | a 20M fork is too short: Δ keeps rising with more breadth steps | **fork extension** (§3.2): N1-P s3501 and C0 s3501 each +40M, read at PR-1's engine; plasticity predicts the extension stays flat, dose that it climbs | ≈ 16 GPU-h + ≈ 180 lane-h |
| **Capacity** | a wider network pilots unseen teams better at equal data | the N4 toy screen (fresh toy at 2× width vs the toy, matched wall clock); a build first (`D_MODEL` is a constant in `src/agents/model/arch_constants.py`) | ≈ 1.5–2 agent-days + ≈ 13–15 GPU-h |
| **The meter** | Kakuna is familiar with our "unfamiliar" panels | `forum216b` (sealed, X54) and the reverse cell; the panel-validity read found no panel shown Kakuna-familiar, and dropping H LOWERS P\*'s score | read for the panels; the open part needs downloads not made |

**The plasticity probe's registered bars** (APPLIED 2026-10-27 in [`plasticity_check_2026-10-26/README.md`](../research_state/measurements/plasticity_check_2026-10-26/README.md) §2, class **NO-LOSS**; the
literature basis was re-fetched there, §7): fit a FIXED new target (P\*'s own greedy labels on one frozen bank of
held-out-team states, through a fresh linear head on the frozen trunk, then end-to-end) for a fixed CPU budget from
each checkpoint. **LOSS** iff P\*'s area under the fit-loss curve is ≥ 1.25× X26 @25M's AND its near-dormant unit
fraction ≥ 2× X26 @25M's; **NO-LOSS** iff the AUC ratio < 1.10; otherwise **UNCLEAR**. The stage confound (a
less-trained network starts at a higher loss) is why the target is fixed and the AUC is read relative to each
checkpoint's own start.

**What the evidence for resolution is, pooled honestly.** The selected four (D1 +2.10, D3 +1.27, X52 unpractised
+1.3, λ = 0.25 +1.50) pool to +1.65 pp (SE 0.97); the six in scope (adding λ = 1 −1.50 and X26c +0.08) to **+0.91 pp
(SE 0.81, [−0.67, +2.49])** (`pgain_rows.log` §2). The levers differ, so neither pool estimates one effect: the "+1–2
pp pattern" exists only after selection. **Resolution's evidence is PR-1's registered family estimand, defined before
any of its games**; nothing pooled after the fact is cited as support.

**One fact cuts against capacity, weakly.** The 1.42M toy T\* drops home72 → F half as far as P\* on the logit scale
(0.14 vs 0.28, NOT DETECTED; X51). That is what capacity-as-memorisation predicts, not capacity-as-generalisation.
Neither reading is supported at that resolution.

---

## 3. The levers

### 3.1 PR-1 — the precision read (zero GPU)

**The question.** Do the breadth levers tried at the plateau, read together, move F against Kakuna by a useful amount
(≥ +1.5 pp), or is "breadth at the plateau, at 20M" worth < 1.5 pp?

| item | the design |
|---|---|
| **arms** | the banked finals of **D1** (s3501–s3505), **D3** (s3501–s3503, + s3504–s3505 if look 2 has trained by registration), **N1-P** (s3501–s3503, + look 2's seeds likewise) and **C0** (s3501–s3505). The seed set is fixed at registration. X52 is excluded (practice on 36 teams: another estimand, 2 seeds); the belief arms are excluded (not breadth) |
| **estimand** | Δ_B = mean over the three breadth arms of (F_arm − F_C0), each arm's F the equal-weight mean of its seed finals: "breadth at the plateau, any corpus, 20M" |
| **meter** | F only: Kakuna ckpt 34, greedy vs greedy, CPU, role-balanced 20-game units, the D3 / N1-P regime. F is the one panel unhazarded for N1-P (H and P72 carry its generator, N1-P PREREG §8). **D1's corpus is forum-sourced, so F is D1's own population** (D1's F +4.0 vs H +2.1, P72 +0.1): the leave-D1-out contrast is a REPORTED line beside the rule |
| **games** | **8,000 F games per breadth arm, 14,000 for C0** (C0 at √3× a breadth arm minimises the family contrast's variance): 1,900 units, 38,000 games, all NEW unit seeds. No banked game is imported, and the in-flight reads' games are REPORTED beside it, never pooled (they were seen before PR-1 was registered, and they are why it runs) |
| **ONE engine** | every game at ONE playing commit held in a pinned worktree for the read's life (`main.reads`' `code.json`; `repin` only onto a byte-identical play path); each unit records `server_version` and the Rust sim binary's sha256, and the read is VOID if two stamps appear. The queue interleaves the arms unit by unit (unit seeds outermost), so a stop at any point leaves the arms within one unit of their planned shares. That stamp becomes **the tree's eval engine**: every T2 read plays at it, so PR-1's C0 is banked for them |
| **interval** | normal, Δ̂_B ± 1.96 × SE, widened by AM-5's √max(1, φ̂), with φ̂ the mean of the four arms' seed-dispersion ratios (`main.x35.d1`'s estimator). A team-cluster bootstrap over F's 72 teams is REPORTED beside it (review m7) |
| **classes** (rule 8: a bound within 1e-9 of a threshold ⇒ UNDECIDED) | **BREADTH-GAIN** iff LB > 0. **KILL** iff UB < +1.5 pp (LOSS iff UB < 0, a sub-case). Otherwise **UNDECIDED**. GAIN and KILL cannot both hold at the registered games (half-width ≥ 1.1 pp) |
| **interim** | ONE binding futility look at half the games: KILL-STOP iff the half-sample UB < +1.5 pp. No efficacy stop |
| **per-arm lines** | REPORTED, never decide (three contrasts, one estimand). Each arm's interval at 8,000 vs 14,000 is ≈ ±1.9 pp |
| **cost** | 0 GPU-h; **38,000 games = 165–238 lane-h** (breadth arms 35–50 each, C0 61–88), **≈ 7–10 days at `--concurrency 1`**; expected ≈ 29,000–31,000 games (126–181 lane-h) under a null; ≈ 1 agent-day (a pooled-family rule in `main.reads`, the spec, the registration) |

**Power** ([`power_precision_read.py`](../research_state/measurements/strategy_revision_2026-10-26/power_precision_read.py)
→ `.log`, 200,000 draws a cell, base F 0.49):

| true family mean | σ_s = 0: GAIN / KILL | σ_s = 0.5 pp | σ_s = 1 pp | E[games], σ_s = 0 |
|---|---|---|---|---:|
| 0 | 0.021 / 0.789 | 0.027 / 0.738 | 0.038 / 0.609 | 29,146 |
| +0.5 | 0.130 / 0.463 | 0.137 / 0.434 | 0.131 / 0.360 | 33,650 |
| +1.0 | 0.418 / 0.164 | 0.392 / 0.169 | 0.316 / 0.163 | 36,468 |
| **+1.5** | **0.764 / 0.035** | **0.709 / 0.042** | **0.566 / 0.055** | 37,610 |
| +2.0 | 0.948 / 0.004 | 0.913 / 0.007 | 0.793 / 0.013 | 37,934 |

- **Heterogeneous effects give the same numbers** (D1 +2 / D3 +1 / N1-P 0 ≡ +1.0 each): the estimand is the mean.
- **σ_s is the read's one unknown.** "Fork seeds overdispersed by ≈ 0" (D1's five, φ̂ 0.77; D3's three, φ̂ 0.60) was
  measured at 420–1,200 games a seed, where a σ_s of 1 pp is hard to see. At σ_s = 1 pp, power at +1.5 falls to 0.57
  and false GAIN rises to 0.038 (φ̂ is estimated on df 16). PR-1 measures σ_s far better (1,600–2,800 games a seed);
  if its φ̂ implies σ_s ≥ 0.75 pp, more SEEDS (GPU) are the only remedy, and the record says so.
- **Cheaper variants:** 3 × 6,000 + C0 10,000 (28,000 games, 122–175 lane-h) detects +1.5 with 0.62 (σ_s 0). With D3
  and N1-P at 3 seeds the numbers barely move (0.756). Pooling is what makes it affordable: one arm read alone at
  17,400 vs 17,400 detects +1.5 with 0.74 for 34,800 games, so the review's per-arm design (D1, D3 and N1-P each vs
  C0) is ≈ 70,000 games, **≈ 300–435 lane-h, ≈ 76–109 an arm**.
- **Outcome odds** (MY JUDGEMENT: the six-estimate pool above used as a prior, which it is not): BREADTH-GAIN ≈ 0.42,
  KILL ≈ 0.33, UNDECIDED ≈ 0.25.

**What each class licenses.**
- **BREADTH-GAIN** ⇒ the stack fork (§3.2 row 3) is licensed, read by the PR rule; the fork extension (dose) rises in
  rank; X53-N1's prior rises (its own trigger is still N1-P's GO).
- **KILL** ⇒ "breadth at the plateau, ≤ 20M, at these corpora" is retired for goal (b) in this era: no stack fork, no
  N3, no further plateau breadth fork. Fresh breadth stays open only through plasticity (N5-S, or X53 behind its σ
  gate). The owner is asked at once unless the plasticity probe reads LOSS (§5).
- **UNDECIDED** ⇒ no breadth fork; the owner is asked at the end of T1 with the number (more games are cheap on a CPU
  box).

**Making it affordable** (in order of preference):
1. **Pool** (done above): ≈ 38,000 games instead of ≈ 70,000.
2. **A rented CPU box, on the owner's approval.** The cloud plan prices GPU boxes only; a CPU-only 32-vCPU instance
   running K parallel units at D1 look 2's uncontended unit rate (3–5 min a 20-game unit) would play the 38,000 games
   in ≈ 8–24 box-h for ≈ $5–25 (UNVERIFIED: no rented CPU has played a Kakuna game; price MY JUDGEMENT). It needs the
   Metamon peer on the box (Kakuna's weights by the pinned hub revision and sha256, public), the 18 finals uploaded
   (a scope change like approval (b) of the cloud plan, `cloud_gpu_plan.md` §6), and the public repo (F's teams are
   committed). No human replay data. **All arms play on the box** (the engine rule); a few units re-played locally
   game for game are REPORTED as the cross-host guard.
3. **`--concurrency 2`** locally (UNVERIFIED gain, `reads.md` §3).
4. **Schedule it behind the current reads** (N1-P look 1 ≈ 60 h, D3 look 2, X60, X59-S's reads): it starts ≈ 1 week
   out and takes ≈ 7–10 days.

### 3.2 The ranked levers

**How to read the table.**
- "P(GAIN)" is the probability of a GAIN on the row's read, integrated over its GUESS (read as a normal with the
  80 % interval's width / 2.563 as SD; [`pgain_rows.py`](../research_state/measurements/strategy_revision_2026-10-26/pgain_rows.py)).
  The two columns are D1's rule (1,200 F games a seed final, σ_s 0) and the **PR rule** (17,400 F games an arm at
  the tree's engine, 5 seeds; sized for +1.5 pp at ≈ 0.74), the rule every T2 fork is now read by.
- Costs are to the FINAL read, not look 1. Lane-h at the PR rule is 17,400 arm games + a 3,400-game C0 top-up at the
  tree's engine (≈ 107 lane-h).
- The GUESSES are conditional on all three misses and are lower than the survey's where a miss bears on them.
- **Six of the old eight rows were one bet** (team breadth). They are now ONE breadth bet whose form PR-1 chooses
  (rows 3, 5–7 and X53); they are not independent chances to compound.

| rank | lever | escapes / tests | GUESS (80 %) | P(GAIN) D1 / PR | GPU-h | lane-h | agent-days | per resource-day (PR) | gate |
|---|---|---|---|---|---:|---:|---:|---:|---|
| **1** | **PR-1** (§3.1) | resolution | — (an information read) | — | 0 | 165–238 | ≈ 1 | — | none: T1's first item |
| ~~2~~ | **N2: set-channel noise — DROPPED 2026-10-26** (its test read COMPOSITIONAL, R = 1.19 [1.07, 1.33]; the row is kept for the record). In training only, each own mon's set fields are masked with ONE registered probability p (no tuning: a second p is a second lever) | identification | +1.5 (0, +3) | 0.41 / **0.64** | 20.5 | ≈ 107 | ≈ 1.5 | **≈ 0.12** | the exact-set identity test (zero GPU), REGISTERED at [`n2_identity_test_2026-10-26/PREREG.md`](../research_state/measurements/n2_identity_test_2026-10-26/PREREG.md) (`98bcb114`): KEYS-ON-SETS (R ≥ 1.5, CI lower > 1.0) funds N2; COMPOSITIONAL (CI upper < 1.5) drops it; INCONCLUSIVE otherwise |
| **3** | **The breadth stack** (breadth bet, plateau form). One union corpus of `w_forum_v1@ABC`, `d3_setmut_v1` and `n1_proc_v1` (panel exclusion re-derived on the union) | resolution (on GAIN) | +1.5 (0, +3.5) after PR-1 GAIN | 0.42 / **0.62** | 20.5 | ≈ 107 | 0.5–1 | **≈ 0.12** | **PR-1 BREADTH-GAIN** only. P is co-primary (D1's −1.6 pp may compound). FINDING 6: no current panel is unhazarded for the union |
| 4 | **H-TEAMS (X60).** Human ladder teams in the pool, the human-data toggle | breadth, human sets | +1.0 (−1, +4) | 0.35 / 0.49 | 12–20 | ≈ 107 | 2–3 | ≈ 0.09 | X59-S not closing human data; X60 READY; LOCAL only |
| 5 | **Specialist distillation.** Practise K ≈ 100 human-built teams that are in no panel (≈ 0.6 GPU-h a team, X52's regime), then distil the K specialists into the generalist (policy distillation, 5 seeds) | the familiarity premium directly | +1.5 (−1, +4) | 0.44 / 0.59 | ≈ 60 + 20 | ≈ 107 | ≈ 3 | ≈ 0.07 | X52's practised gain stands (+3.2 [−0.2, +6.8]); its teams carry the F-adjacency hazard of any forum source |
| 6 | **Fork extension (dose vs plasticity).** N1-P s3501 and C0 s3501 each +40M at their pins; read at the tree's engine | dose | +1.0 (−1, +3) on the extended Δ | 0.32 / 0.49 | ≈ 16 | ≈ 180 (two new checkpoints) | ≈ 0.3 | ≈ 0.06 | N1-P UNDECIDED with Δ̂ > 0, or PR-1 BREADTH-GAIN; mostly diagnostic |
| 7 | **N5-S: an offline breadth student FROM SCRATCH.** A CPU self-play corpus by P\* and its archive on ≥ 10⁴ teams; a FRESH student (X59-S's students are fresh: `main.fresh_checkpoint.build_fresh_model`) trained by advantage-filtered BC + a critic, against the same student on a 719-team corpus | plasticity (the only cheap fresh-breadth test) | +1.0 (−2, +5) on the student contrast | 0.38 / 0.49 | ≈ 8 | ≈ 60–100 corpus (UNVERIFIED rate) + ≈ 107 | 3–4 | ≈ 0.06 | X59-S's students trained cleanly; a measured CPU forward rate for P\* at corpus scale. The from-P\* student is NOT ranked: it escapes none of the five explanations |
| 8 | **N3: a regret-scored team curriculum** (PLR over teams) | breadth, allocation | +0.5 (0, +3) | 0.19 / 0.33 | 20.5 | ≈ 107 | 2–2.5 | ≈ 0.06 | its zero-GPU probe (ρ ≥ 0.2); runs only on top of a breadth corpus that has read positive (PR-1 GAIN) |
| 9 | **Batch × 2 with breadth.** N1-P's argv at twice the effective batch (65,536 → 131,072) against N1-P itself (banked at the tree's engine by PR-1) | breadth's gradient noise | +0.5 (−1, +3) | 0.23 / 0.37 | ≈ 20.5 (throughput UNVERIFIED) | ≈ 107 | ≈ 0.5 | ≈ 0.07 | PR-1 not KILL. MY JUDGEMENT (OpenAI Five and AlphaStar scaled batch with diversity; not re-fetched) |
| 10 | **N4: width.** The toy screen first, then production with a fresh control (CF, shared with X53) | capacity | toy: a screen; production +1.5 (−1, +4) | — | toy 13–15; production 40–60 + CF | 30–54 | 1.5–2 | — | the plasticity probe NO-LOSS (the capacity branch) |
| 11 | **X53-N1 without N1-P's GO** (fresh breadth pair, behind its σ gate) | plasticity | +1.5 (−1, +5) at 100M | ≈ 0.2 over its GUESS on its own chain (0.30 at +3 pp, σ 1.5) | ≈ 46–72 expected | ≈ 130–185 with its σ proxy | ≈ 1 | ≈ 0.02 | plasticity LOSS **and** PR-1 not KILL **and** X53's σ gate passes |
| 12 | **The scale bet:** N4 × N1 from scratch to the plateau, two seeds | capacity + plasticity | +3 (−1, +8) | — | ≈ 150–250 | ≈ 150 | ≈ 3 | — | the N4 toy GO; a second card; the owner's word |

**Listed, not ranked, with the reason:**
- **Option T, the toy fresh pair** (`main.x61.arms`, ≈ 13 toy GPU-h): the cheap screen if N1-P is UNDECIDED, as X53
  §4 says; not a GAIN path, because the toy's familiarity premium is about half P\*'s and its transfer to production
  is uncalibrated.
- **Reward shaping:** no mechanism tied to piloting an unfamiliar team.
- **Opponent-pool diversity at the fresh start:** inside X53's CF and N1F by construction.
- **More seeds of what worked:** that is PR-1.
- **Test-time adaptation** (practise the handed team, ≈ 0.6 GPU-h): a deployment REGIME, never the generalist's
  number; an owner option at the exit (§5). Row 5 turns it into a training lever.
- **A strategy latent (AlphaStar's z):** no Pokémon evidence, a multi-day build.
- **High-temperature collection:** needs an off-policy correction in PPO; N5-S's corpus reaches it.

**GPU-free levers**, ranked by CPU and agent-days:
- **Search, E2 → E1 → E4** (`design_search_next.md` §3). At P\*, L1's anchored search reads +0.0001 [−0.0097, +0.0101]
  per decision; E1-0's real-bot referee reads UNDECIDED. +4.6 pp in games needs only ≈ +0.0018 per decision (the
  performance-difference bridge). **E2 (the bridge):** ≈ 0.5 agent-day and 25–35 lane-h. E4, a live player against
  Kakuna, is gated on E1–E3 and costs ≈ 300–470 lane-h. Expected for goal (b): +0.5 (−1, +2.5) (GUESS).
- **An ensemble of our checkpoints at play:** +0.5 (0, +2), ≈ 1 agent-day and ≈ 10 lane-h; its members trained on
  the same teams, so it cannot fix piloting breadth.

**Honest uncertainty.**
- The P(GAIN) columns are near-ties (0.19–0.44 on D1's rule, 0.33–0.64 on the PR rule): **the ranking separates
  costs, not effects.** Reading forks by the PR rule raises every P(GAIN) by ≈ 0.15–0.20 and costs ≈ 3× the CPU
  of D1's rule; it is the price of not retiring real +1–1.5 pp levers (D1's rule calls a true +1.0 NULL 0.32–0.47).
- No row's point estimate alone reaches +4.3 pp except the scale bet's. The plausible route to goal (b) is two or
  three +1–2 pp gains read together, which is what PR-1 tests for the breadth family.

---

## 4. The decision tree

```
T0  the three reads land (D3 look 2 · N1-P look 1 [+ look 2] · X59-S)
 ├─ any GAIN ─────────► its own licence (N1-P ⇒ X53-N1; D3 ⇒ adoption path; X59-S ⇒ X60). This doc does not apply.
 └─ none ─────────────► T1
T1  THE ZERO-GPU STAGE (the GPU keeps its queue). At its START the owner is asked one question: a CPU box for PR-1?
      a. PR-1, the precision read (§3.1)          ≈ 1 box-day, or ≈ 7–10 days of the local anchor lane
      b. N2's exact-set identity test: RAN 2026-10-26, COMPOSITIONAL ⇒ N2 dropped
      c. N3's PLR-signal probe                     ≈ 0.5 agent-day
      d. the plasticity probe (§2's bars)           RAN 2026-10-27: NO-LOSS (≈ 2 slot-h, ≈ 6 CPU-h)
      e. search's E2 bridge read                   ≈ 0.5 agent-day, 25–35 lane-h
      f. the piloting-gap analysis, if it has landed (routing below)
T1 END ⇒ THE OWNER (§5), with PR-1's class and d's class in hand
T2  CAPPED FORKS (each read by the PR rule at the tree's engine; cap ≈ 40 GPU-h and ≈ 250 lane-h without a GAIN)
      PR-1 BREADTH-GAIN ─► the STACK fork (rank 3); then the extension (rank 6); N3 (rank 8) if c passes
      (b read COMPOSITIONAL: no N2 flag or fork)
      X59-S UNCLEAR ─────► H-TEAMS once X60 is READY (rank 4; local)
      PR-1 not KILL ─────► batch × 2 (rank 9) only if the cap has room
T3  THE BRANCH ON d
      plasticity LOSS ──► N5-S (rank 7) first (≈ 8 GPU-h); X53-N1 (rank 11) only behind its σ gate and PR-1 not KILL
      NO-LOSS ──────────► the N4 toy screen (rank 10); GO ⇒ production width with a fresh control   ◄ TAKEN (d read 10-27)
      UNCLEAR ──────────► the behavioural plasticity complement (§2, ≈ 8–16 GPU-h) before either
T4  E1 → E4 (search), only if E2 shows the bridge holds and an arm predicts ≥ +4.6 pp. CPU only
T5  THE SECOND ESCALATION: the T2 cap reached without a GAIN, or every branch exhausted ⇒ the owner (§5)
```

**Routing on the piloting-gap analysis** (by outcome class; the analysis owns its numbers):
- the gap concentrates in identifiable decision classes (switching, set-up, item-dependent lines) ⇒ N3 with a
  targeted score, and a targeted D3-style corpus for those mons, move first;
- it concentrates in team identity (familiar sets read as a signature) ⇒ N2's test already argues against a
  set-keyed signature (COMPOSITIONAL); N2 re-opens only with a probe that reaches revealed / active mons or the
  battle's course, which the one-mon bench re-draw cannot;
- it is spread evenly over decisions ⇒ breadth and capacity carry the hope; PR-1's class decides which;
- it localises to the critic (value errors on unfamiliar teams, not policy errors) ⇒ the amortisation question in
  `design_beat_metamon_path.md` §4 is re-opened before any T2 fork.

**Why this order.** T1 costs no GPU, can kill N2, N3 and the whole plateau breadth bet outright (it dropped N2 on
2026-10-26), and points T3 one way. T2's forks run in the X35 frame against a C0 that PR-1 has already banked at the tree's engine. T3 is the
expensive split and waits for the probe that chooses the side.

---

## 5. The owner escalation: at the end of T1, with an information trigger

**Asked at the START of T1:** "approve a rented CPU box (≈ $5–25, the 18 fork finals uploaded, no human data) for
PR-1?" A yes shortens T1 from ≈ 1–2 weeks to ≈ 2–3 days.

**Asked at the END of T1** (the trigger, deterministic on the registered classes):
- **PR-1 KILL and the plasticity probe not LOSS** ⇒ escalate AT ONCE: no cheap breadth lever remains, and what is
  left is capacity or scale (N2 was dropped 2026-10-26).
- **PR-1 KILL and plasticity LOSS** ⇒ escalate with the fresh-breadth costs (N5-S ≈ 8 GPU-h; X53-N1 behind its σ gate,
  ≈ 46–72 GPU-h expected); T3's N5-S may run meanwhile.
- **PR-1 BREADTH-GAIN or UNDECIDED** ⇒ T2 runs under its cap; the owner is told the class and the plan, and asked
  again at the cap (T5).

**What the owner is asked** (either point):
1. **Re-scope goal (b)?** The deployment-shaped headline L (we bring our teams, Kakuna brings H) already reads 0.630
   [0.573, 0.686], and goal (a) is met. Goal (b) is the harder, team-neutral question (`design_beat_metamon_path.md`
   §1).
2. **Adopt test-time adaptation as the deployed regime?** (≈ 0.6 GPU-h per team; a regime, never the generalist's
   number.)
3. **Fund the scale bet** (≈ 150–250 GPU-h, rank 12) with a second or bought card? A card has a lead time, which is
   why the question comes at the end of T1 and not after ≈ 90 GPU-h.
4. **Or return the GPU** to the other standing goals (search, the AlphaStar-style league), leaving goal (b) open with
   the evidence above.

---

## 6. The second GPU, and the CPU, in this plan

- **A second card is a rate, not a lever.** A rented 5090 costs ≈ $0.38 per local GPU-h of work at ≈ 1.65× our card
  ([`../ops/cloud_gpu_plan.md`](../ops/cloud_gpu_plan.md) §3). T2's forks go on the cloud only behind the cloud
  control C0v (the card rule §1.2); fresh pairs (X53-N1, a wide arm) bring their own control and are the natural
  cloud workload; H-TEAMS and every human-data arm stay local.
- **The CPU is the binding lane for this plan** (the cloud plan's FINDING, made concrete): PR-1 is 165–238 lane-h,
  each PR-rule fork read ≈ 107, the extension ≈ 180, E4 300–470. On rule 5b's three agent cores that is weeks. A
  rented CPU box (§3.1) is the cheapest lever on the whole tree's wall time, and every PR-rule read could use it on
  the same approval.

---

## 7. What this plan does not do

- It never pools after the fact: the only pooled breadth claim is PR-1's registered estimand.
- It does not rank by expected pp: a lever's decision value is its P(GAIN) on its own rule, per full cost.
- It does not read any fork by D1's ±3 pp NULL band: that band retires a true +1.0 pp lever 0.32–0.47 of the time.

---

## 8. FINDINGS (standing rule 7)

1. **The scenario's probability** is 0.58 (independent) to 0.68 (D3 and N1-P perfectly correlated).
2. **The piloting-gap analysis had not landed** (no `piloting_gap_forensics_2026-10-26/` at `6fca2c06`). §4 routes on
   its outcome classes only.
3. **The engine-mix audit had not landed** (no engine table in `designs/tools/reads.md`). PR-1 follows REVIEW F's
   F-M2 fix pattern (one harness commit, the transport identity recorded per unit); `main.reads.assemble.
   ANCHOR_REGIME_KEYS` still omits `server_version`, so the one-stamp check is the read's own until the audit lands.
4. **D3's input to P(all miss) should be its look-2 power at its look-1 posterior**; not recomputed here.
5. **PR-1's power rests on σ_s ≤ ≈ 0.5 pp**, which "≈ 0" at 420–1,200 games a seed does not establish (§3.1).
6. **No current panel is unhazarded for the stack's union corpus**: F carries D1's forum source, H and P72 carry N1-P's
   generator. The stack's registration needs a panel decision (or a REPORTED-only F) before it can be read.
7. **The CPU-box throughput and price are UNVERIFIED**; no rented CPU has played a Kakuna game.
8. **Every P(GAIN) integrates a GUESS** read as a normal; the numbers order rows, they do not predict outcomes.
9. **The plasticity probe RAN (2026-10-27): NO-LOSS.** Its registered target (P\*'s greedy labels through a linear
   head) is weakly sensitive and favours P\* by construction; the architecture-neutral target carries the trend, and
   it sits on the 1.10 line (1.101–1.125 across seeds). It measures supervised refitting, not PPO under a domain
   shift, which only the behavioural complement reads (the probe's FINDINGS F2, F5).
10. **N5-S's corpus cost rests on an UNVERIFIED CPU forward rate** for P\* at corpus scale (X59-S's ≈ 2.6 ms per
    forward is the nearest figure).
11. This unit ran no game and used no GPU. Its Monte Carlos ran on the agent cores at nice 19 (< 1 min each).

---

## 9. RESPONSE to the review (`6fca2c06`)

| finding | disposition | what changed, or why not |
|---|---|---|
| **B1** (BLOCKER): no step resolves +1–1.5 pp; the zero-GPU resolution read is missing | **FIXED** | PR-1 is T1's first item (§3.1): the pooled breadth-family estimand the review proposed, at 38,000 F games (0.76 at +1.5 pp, σ_s 0), one engine, new unit seeds, a futility-only interim, costed at the MEASURED rate (165–238 lane-h, not the review's 145 at 360 games a lane-h). Every T2 fork is now read by the PR rule (sized for +1.5 pp), not D1's |
| **M6**: selected evidence; the belief channel "closed" and evidence at once | **FIXED** | §2 prints both pools (+1.65 SE 0.97 selected; +0.91 SE 0.81 all six) and says neither estimates one effect; resolution's evidence is PR-1's registered estimand; §1 calls the belief channel NOT DETECTED at ±4–5 pp, not ranked |
| **M7.1**: six rows are one bet | **FIXED** | one breadth bet whose form PR-1 chooses (rows 3, 5–7, 11); §3.2 says so |
| **M7.2**: N5 from P\* escapes none of the explanations; X59-S's students are fresh | **FIXED** | N5-S (from scratch, against a 719-team student) is the ranked form, re-guessed at +1.0 (−2, +5); the from-P\* form is not ranked |
| **M7.3**: expected pp is the wrong numerator | **FIXED** | the table ranks by P(GAIN) on each row's rule per resource-day (`pgain_rows.py`) |
| **M7.4**: look-1 costs; CPU left out | **FIXED** | costs are to the final read; lane-h is a column at the measured rate and enters the ratio |
| **M8**: the escalation is where the budget runs out | **FIXED** | the owner is asked at the end of T1 with a deterministic trigger (§5); T2 is capped at ≈ 40 GPU-h AND ≈ 250 lane-h; the CPU-box question comes at T1's start |
| **M9**: missing options | **FIXED** | rows for the fork extension (dose; X53 §4's branch now has a home), batch × 2, specialist distillation; Option T, reward shaping and pool diversity listed with reasons; "dose" is the fifth explanation |
| **m9**: P(all miss) bookkeeping | **FIXED in part** | the range 0.58–0.68 and the outcome-type mix are stated; D3's look-2 power at its posterior is not recomputed (FINDING 4) |
| **m10**: the plasticity probe's stage confound | **FIXED** | a fixed new target, the AUC relative to each checkpoint's start, registered bars, and the behavioural complement (§2) |
| **m11**: N2's p and its cost | **FIXED** | ONE registered p, no tuning; the cost is the final's |
| **m12**: the exit counts GPU-h only | **FIXED** | the T2 cap counts lane-h; §6 names the CPU as the binding lane |
| the review's sizing "≈ 17,400 F games per arm" | **ADOPTED for single-arm reads, REPLACED for PR-1** | pooling three arms against a √3×-weighted C0 reaches 0.76 at +1.5 for 38,000 games where per-arm reads need ≈ 70,000 |

---

## 10. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT)** (research-strategy worker, orchestrator brief) | a contingency plan for D3, N1-P and X59-S all missing a GAIN: four explanations and the cheapest read separating each; a ranking by expected pp per GPU-h (N5, N2, the stack first); a zero-GPU week, cheap forks, a plasticity branch, an owner escalation at ≈ 90 GPU-h | another single lever read alone; more plain training; FACING levers; the scale bet first; test-time adaptation as goal (b)'s number | `design_beat_metamon_path.md` §0–§6; `sota_survey_2026-10-26.md` §5; the D3, N1-P and X59-S PREREGs; `design_search_next.md` §3; `cloud_gpu_plan.md` |
| 2026-10-26 | **REVISED DRAFT after the review** (research-strategy worker, orchestrator brief; docs + CPU Monte Carlos) | PR-1, the zero-GPU precision read, as T1's first item (D1 + D3 + N1-P vs C0 pooled on F, 38,000 games at one engine stamp; GAIN / KILL at +1.5 pp; futility interim); five explanations (dose added); one breadth bet chosen by PR-1; rank by P(GAIN) per resource-day with lane-h counted; T2 forks read by the PR rule; N5 from scratch; new rows (extension, batch × 2, distillation); the owner asked at T1's start (CPU box) and end (information trigger) | per-arm precision reads (≈ 70,000 games); D1's rule for T2 forks (retires real +1 pp levers); the ≈ 90 GPU-h budget exit; N5 from P\*; ranking by expected pp | [`REVIEW.md`](../research_state/measurements/strategy_review_2026-10-26/REVIEW.md) (`6fca2c06`); [`strategy_revision_2026-10-26/`](../research_state/measurements/strategy_revision_2026-10-26/) (`power_precision_read`, `pgain_rows`); D3 look-1 README FINDING 5 and `reads.md` §3 (the measured lane rate); D1 read README (F per panel) |
| 2026-10-26 | **N2 DROPPED by its registered exact-set identity test** (N2 identity-test worker, orchestrator brief; CPU only) | COMPOSITIONAL: a Smogon re-draw of one never-seen bench set moves P\*'s policy 1.19× [1.07, 1.33] as much on a pool team as on an unfamiliar one (pattern-standardised KL, 7,260 / 24,384 units, 48 / 143 games); erasing our six mons' identity embeddings hurts both alike (1.07 [0.91, 1.25]); mechanics-preserving edits are invisible. The interval excludes 1.0 (a small familiarity premium, ≈ the generalisation review's +19 %), so the verdict rests on the PREREG's declared rule (upper < 1.5), not on the older literal KILL clause | funding the N2 flag (the premium it could remove is ≈ 0.2 of one set change's effect); reading the premium as memorisation | [`n2_identity_test_2026-10-26/README.md`](../research_state/measurements/n2_identity_test_2026-10-26/README.md) (PREREG `98bcb1146`); `piloting_gap_forensics_2026-10-26/` (team context, not set knowledge) |
| 2026-10-27 | **The plasticity probe READ: NO-LOSS ⇒ T3 takes the capacity branch** (plasticity-check worker, orchestrator brief; CPU only) | the §2 bars applied to X26 @25M … X26c @480M, C0 and two fresh inits on 11,511 held-out-team states: P\*'s end-to-end refit AUC on its own greedy labels 0.986× @25M's (seed pairs 0.984–0.987; NO-LOSS < 1.10), near-dormant 0.78× (LOSS needs ≥ 2×); on a neutral random-network target the trunk refits slower than @25M (1.11×, UNCLEAR by the same bars) but the slow-down is paid by 175M and does not grow across the plateau (100M 0.576 → P\* 0.598 → 480M 0.602); the one structure still degrading is the policy MLP's tanh saturation | UNCLEAR (the behavioural complement first) on the neutral target's 1.11, which lies inside its seed noise of the line; the registered class decides | [`plasticity_check_2026-10-26/README.md`](../research_state/measurements/plasticity_check_2026-10-26/README.md) (results, `summarize.py`) |
