# X65 ExIt-0: does a P\* student distilled from S′'s banked choices beat P\* against the real Kakuna on teams it never saw? — PREREG (2026-11-05)

Written and committed BEFORE any bank row is built, any student trains or any game is played. Design:
[`design_x65_exit0_search_distillation.md`](../../../endstate/design_x65_exit0_search_distillation.md).

**Status.**
- The question, the classes, the folds, the recipe, the deals, the statistic and every bar are frozen here.
- Before the first student trains, three things land. Each is a BLIND amendment, written before any student logit or
  game exists:
  - builds B1–B5 (§9);
  - one adversarial review, every finding disposed;
  - a pin amendment naming the code and bank commit and the bank's manifest sha256.
- Nothing in an amendment may move a bar, a fold, a deal, z, β, the learning-rate grid, the epoch budget or the class
  rule.

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…91ed`).
- S′ = `r0s4_rolls@10` (8 worlds, `dice="rolls"`, TRUNC-4-K8, piKL λ 0.015). Kakuna = Metamon's Kakuna, checkpoint 34,
  greedy, peer pack `gen3_peer_showdown_pack_v1`.
- U = H (`heldout72_hpfix`), F (`forum72`), p72. E4 = the engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.
- S′-LIVE = `../sprime_live_play_2026-11-01/` (record `fcfe2575e`; state `~/gen3ai_archive/sprime_live_play_2026-11-01/`).
- y = a game's result for us (1 / ½ / 0). D = y(student) − y(P\* raw) for one deal.
- π′ = softmax over the legal actions of (log π_P\* + Q̂ / λ), λ = 0.015.
- LB / UB = the lower / upper bound of the stated interval. GUESS = an unmeasured effect. ESTIMATE = a cost from
  measured rates.

## 1. The question and the classes

**Does a P\* student, trained offline toward S′'s piKL targets on two panels' banked decisions (with a KL anchor to
P\*), win more games against Kakuna than P\* on the third panel's deals, whose teams it never saw?** The student plays
greedy with no search.

| class | rule (§6) | licenses | does NOT license |
|---|---|---|---|
| **NOT-DELIVERED** | the delivery gate's LB(Δagree) ≤ 0 (§5); no Kakuna game is played | "At this recipe the students did not learn S′'s choices on unseen teams." X60-T is the GPU's next candidate | anything about larger budgets, honest labels or in-loop distillation |
| **GO** | DELIVERED and LB(G_X) > 0: at look 1 with z = 2.576, or at look 2 with z = 2.0 | "A network distilled from S′'s leak-inclusive choices beats P\* against Kakuna on unseen teams, with no search at play time." Registering ExIt-0H (honest labels) and ExIt-1 (PPO + auxiliary KL, ≈ 35 GPU-h); naming the students as an X54 candidate on X54's own terms. The §7 reported checks run | "goal (b) met"; adoption as production; any claim about S′-HONEST; pooling with S′-LIVE's G_U |
| **NULL** | DELIVERED, not GO, and UB(G_X, 95 %) < +2.0 pp at look 1 or at look 2 | "One supervised distillation step on ≈ 65k S′ labels keeps less than 2 pp on unseen teams." ExIt-1 is not funded | closing ExIt (more labels, honest labels, an in-loop form, a second iteration) |
| **UNDECIDED** | otherwise, after look 2 | nothing; no extension without a new registration | — |

- **Look 1** decides GO or NULL if either holds; otherwise look 2 plays.
- **A bound within 0.0005 of its bar is never a crossing** (root rule 8): that class does not hold at that look, and
  the record is flagged `boundary`.
- **GO is checked before NULL**; both cannot hold, since GO needs LB > 0 and NULL needs UB < +2 pp.
- **Not a use of S′-LIVE's licence.** S′-LIVE's GO-LEAKY licenses only a leak-honest replication and no training
  change. This read is its own registration under the orchestrator's brief of 2026-11-05. It trains research students
  and changes no production training, baseline or recipe.

## 2. The label bank (B1; outcome-blind)

- **Source.** S′-LIVE's COUNTED searched units (its `record.json` counted set, final attempts only; never a
  `.partial.<epoch>` dir). The builder reads `search_decisions.jsonl` and `capture/*.json` (START and commands) alone.
  It never reads `games.jsonl`, `summary.json`, a capture's result line or the record's outcomes.
- **A label row** is one record with `searched: true`. It holds:
  - the observation row: our side's row rebuilt from the capture by replaying the honest input log in `honest_log`'s
    (update, side, arrival) order through `main.value_search.worlds.prepare_root`;
  - the legal mask: the keys of `q`;
  - π′ from `q` and `log_pi` at λ = 0.015;
  - the keys: panel, unit seed, deal, `key`, and the leak block.
- **The identity guard (a GIGO check).** Every label row's rebuilt frame must hash to its `frame_sha256`.
  - A mismatch is DROPPED and counted.
  - More than 0.5 % of a panel's label rows dropped VOIDs the bank. The fix is B1's, followed by a blind amendment.
  - Measured before registration ([`frame_probe.json`](frame_probe.json)): 976 / 977 bit-exact on 30 games with a
    naive capture prefix. The one miss is the ordering case that B1's ordered rebuild exists to remove.
- **Exclusions** (`searched: false` past turn 10, no consistent world) carry no label.
- **Anchor rows.** Every decision of our side, both arms, in the TRAINING panels' counted games, rebuilt the same way.
  Their teacher is P\* (computed online from P\*'s bytes). No `frame_sha256` exists for raw decisions, so B1 also
  re-derives the searched arm's anchor rows past turn 10 against their recorded frames as its path check.
- **Counts** ([`bank_count.json`](bank_count.json), all final dirs, before the counted-set filter): H 32,020, F 33,398,
  p72 32,183 searched decisions.

## 3. The folds, the students, the recipe

| student | trains on (labels + anchor rows) | reads on | training seed |
|---|---|---|---|
| S_H | F + p72 | H | 6511 |
| S_F | H + p72 | F | 6512 |
| S_P | H + F | p72 | 6513 |

- **Start:** P\*'s weights, loaded from P\*'s zip. **Architecture:** P\*'s; no growth.
- **Loss:**
  - L = mean over label rows of KL(π′ ‖ π_θ), plus β · the mean over anchor rows of KL(π_P\* ‖ π_θ), with β = 1.0;
  - both KLs run over the legal actions;
  - π_θ is the policy's masked log-softmax (`main.x50_r1.hom.policy_logits_t`).
- **Batch:** 256 label rows + 256 anchor rows, each drawn without replacement per epoch from a plan keyed by the seed.
- **Optimiser:** AdamW (weight decay 1e-5, eps 1e-5); a linear warm-up over 200 steps, then a constant learning rate;
  gradient-norm clip 0.5; fp32; eval mode. Only parameters on the policy-logit path take gradient; the value head's
  own parameters are frozen.
- **The grid and the pick (blind to every read panel):**
  - learning rate ∈ {1e-5, 3e-5}; 4 epochs over the label rows; a checkpoint at every quarter epoch (16 per learning
    rate);
  - the inner-validation set is the TRAINING panels' deals whose `deal` key's sha256 has a first byte ≡ 0 (mod 10).
    Their label and anchor rows are withheld from training;
  - the pick is the checkpoint with the lowest inner-validation L, with ties to the lower learning rate and then the
    earlier checkpoint;
  - the 32 candidates' inner-validation L values are recorded.
- **The export (B3):**
  - Write P\*'s zip with the picked student's policy weights, at the play pin.
  - Its logits must equal the trainer's, bit for bit, on the 64-row compile-parity fixture (CPU, fp32).
  - It must load by the bare loader and play one smoke game against a local peer.
  - Its sha256 is recorded.
- **Device:** CUDA under the orchestrator's lease (`gpu_lease.sh` / `gpu_lock.sh`). ESTIMATE ≈ 0.15 GPU-h a fold,
  from X59-S's measured ≈ 0.4 GPU-h for a 3.6M-row-pass student; this is ≈ 1M row-passes.

## 4. The deals, the arms, the engine

- **Look 1:**
  - **The deals.** Each student plays its read panel's 900 S′-LIVE deals: U unit seeds 300–344, 20 deals a seed,
    `main.anchors --games 20 --deal-game g`. Each is a lone one-game unit against Kakuna, at S′-LIVE's play pin
    `0c76b521e` on E4.
  - **The control.** S′-LIVE's banked `pstar` (raw, greedy) unit on the same deal.
- **G-BANK (before look 1 counts).** Re-play 135 banked raw units at the play pin: for each panel and unit seed, game
  `g = (unit_seed − 300) mod 20`.
  - The protocol (`|t:|` removed, both sides, the START document) must be byte-identical to the banked capture's
    chunks in at least 133 of 135.
  - The measured Kakuna lone-replay divergence is 9 / 2,100.
  - Otherwise the banked control is VOID for look 1, and look 1 plays a fresh raw arm on the same deals (+≈ 22 CPU-h,
    ESTIMATE).
- **Look 2:**
  - **The deals.** U unit seeds **600–674** on each panel: 1,500 deals a panel, 4,500 in all. They are disjoint from
    S′-LIVE (300–344), dlive (200–244) and the X35 frame (< 120). FINDING X65-F3: S′-HONEST must not claim 600–674.
  - **The arms.** Both the student and P\* raw are played fresh, as lone one-game units, at the same pin and engine,
    queued deal by deal (raw, then student).
- **Every counted unit** of the students and of any fresh raw arm must be on E4 and on the recorded model sha256, or
  the read is VOID.
- **Drops.**
  - A deal with one side missing after both attempts is imputed at its worst case: for GO, student 0 and raw 1; for
    NULL, student 1 and raw 0. The class uses the imputation that is adverse to it.
  - More than 0.5 % of a panel's deals dropped at a look VOIDs that look.
  - Look 1's two banked raw drops (S′-LIVE FINDING OP-7) enter as drops.

## 5. The delivery gate (B4; zero Kakuna CPU; read before any game)

- **The decisions.** For student S_X, every label row of panel X (its read panel, never trained on), from S′-LIVE's
  searched decisions.
- **The statistic.** a(student) = 1[argmax π_student = `played`], and a(P\*) = 1[`pi_argmax` = `played`]. Take
  Δagree_X as the mean of a(student) − a(P\*), and pool as the mean over the three panels of Δagree_X.
- **The interval.** Bootstrap by game within panel: B = 4,000, `default_rng(20261105)`. Take the 2.5 % percentile as
  LB.
- **DELIVERED iff LB > 0.**
- **Reported:**
  - Δagree per panel;
  - Δagree split by leak exposure (`attribution_miss` ≥ 1, `rewritten_moves` ≥ 1, `residual` counters, or none);
  - agreement at departures only;
  - the student's agreement with P\* on the read panel's turn < 10 decisions.

## 6. The statistic and the rule `distill_play` (B4)

- **The estimate at a look.** For panel X, Ĝ_X = the mean of D over that look's cumulative deals: look 1's 900;
  look 2's 900 + 1,500. The estimate is Ĝ = (Ĝ_H + Ĝ_F + Ĝ_P) / 3.
- **The SE.** SE = the standard deviation of Ĝ* over B = 4,000 bootstrap replicates. Each replicate resamples, within
  each panel, the unit seeds with replacement and their deals whole (`default_rng(20261106)`; look 2 resamples look 1's
  and look 2's seeds as one pool).
- **The bounds.** LB_z = Ĝ − z·SE (GO: z = 2.576 at look 1, z = 2.0 at look 2). UB = Ĝ + 1.96·SE (NULL at either
  look).
- **Imputation.** Each bound is computed under the imputation adverse to its class (§4).
- **Checking it.** The rule is code in `main.reads`, with constructed-case tests that fail on revert. A `by_hand.py`
  recomputes every panel fraction, bound and the class with its own loop.

## 7. Reported, never decided

- **Each arm's pooled-U score.** Quoted, never as goal (b)'s verdict: X54's sealed `forum216b` owns that.
- **The distilled fraction** Ĝ / G_S′ on look 1's deals, where G_S′ is S′-LIVE's searched − raw on the same deals. It
  is biased low, because S′'s GO was read on these deals.
- **Discordance; game length; the students' V drift** (the mean |V_student − V_P\*| on the read panel's label rows).
- **On GO only** (≈ 37 CPU-h, ESTIMATE):
  - **PLACEBO students** (one per fold). The same recipe, rows and seeds, with each decision's Q̂ permuted among its
    legal actions by `default_rng([6599, row index])`. They play look 1's deals. Attribution: SEARCH-INFO iff the
    student − placebo LB > 0 (95 %), else NOT-SEPARATED.
  - **Goal (a)'s floor:** each student and P\* against Kakuna on home72, 300 deals a student (unit seeds 600–614),
    paired.

## 8. Power and cost

**Power** ([`power.py`](power.py) → [`power.txt`](power.txt), a two-look normal Monte Carlo, 200,000 draws a cell).
The share of deals where the arms differ is d ∈ {0.45, 0.50}; S′-LIVE's was 0.341 with a 10-turn shared prefix.

| true Δ | 0 | +1.0 | +2.0 | +2.3 | +3.0 | +4.6 |
|---|---|---|---|---|---|---|
| P(GO) | 0.026 | 0.22–0.24 | 0.66–0.71 | 0.78–0.82 | 0.95–0.96 | 1.00 |
| P(GO at look 1) | 0.005 | 0.03 | 0.13–0.15 | 0.19–0.21 | 0.36–0.40 | 0.79–0.84 |
| P(NULL) | 0.70–0.74 | 0.26–0.28 | 0.04 | 0.02 | 0.004 | 0 |

The false GO is 0.026 one-sided against a nominal 0.025. GUESS for the true Δ: +1.5 pp (−1, +4); P(GO) ≈ 0.4.

**Cost:**

| item | measured basis | ESTIMATE |
|---|---|---|
| bank (B1) | 0.012 CPU-s a rebuilt row (`frame_probe.json`) × ≈ 98k label + ≈ 150k anchor rows | ≈ 1 CPU-h |
| students | X59-S ≈ 0.4 GPU-h a 3.6M-row-pass student | ≈ 0.5 GPU-h (3 folds × 2 learning rates) |
| delivery gate | CPU forwards of ≈ 98k rows (dlive: ≈ 3.5 ms a row on one CPU thread) | ≈ 0.3 CPU-h |
| G-BANK + look 1 | S′-LIVE's raw units: 22 CPU-h / 2,698 = 29 CPU-s a unit | ≈ 23 CPU-h |
| look 2 | 9,000 units × 29 CPU-s | ≈ 73 CPU-h |
| on GO | placebo 2,700 + home72 1,800 units | ≈ 37 CPU-h |

- **Wall time** at concurrency 6 with no run live: ≈ 5–6 h for look 1 and ≈ 15–18 h for look 2 (GUESS, from a raw
  unit's ≈ 32 s of wall). At concurrency 2 beside a live run, GUESS 3–6× slower.
- **Order:** the orchestrator's. Every look must finish, or its lane be held, before S′-HONEST's live read starts; this
  read never shares the lane with it.

## 9. Builds (before the pin amendment)

| # | build | must fail on revert |
|---|---|---|
| B1 | the bank (§2): an ordered rebuild, the `frame_sha256` guard, anchor rows, fold and inner-validation keys, a manifest with sha256, outcome-blind (a test that refuses a `games.jsonl` read) | the identity guard on a constructed mis-ordered log |
| B2 | the fine-tune (§3), resumable; the 32 candidates' inner-validation values recorded; a refusal of an unregistered bank sha | the inner-validation split never containing a read panel's deal |
| B3 | the export at the play pin (§3) | bit-identity on the fixture |
| B4 | the delivery gate and the rule `distill_play`, with constructed cases for every class and the boundary | each class's constructed case |
| B5 | the read spec (`check-spec` exit 0): G-BANK, three model labels, look 2's seeds | — |

## 10. VOID (any one voids the read or the look named)

- The bank's drop share above 0.5 % on any panel.
- A student whose inner-validation split contains a read panel's deal.
- The export failing bit-identity.
- A counted unit off E4 or off the recorded sha256.
- Drops above 0.5 % of a panel at a look.
- An unfinished deal at a look's close.
- `check-record` not OK.
- `by_hand.py` disagreeing with the rule.

## Reviews

- [`REVIEW.md`](REVIEW.md) (`9b6c0541d`, 2026-11-05): the adversarial review of the PREREG and builds B1–B5 at
  `c9d86ef53`. Verdict TRAIN, no blocker; 3 MAJOR, 12 MINOR. R-M2 is disposed in Amendment 1, every other finding
  in the pin amendment.

## Amendment 1 (2026-11-05, before any student trained, any student logit or any game): ExIt-1 waits for the placebo

Written before any student trained, any student logit existed or any game of this read was played (the bank is
building, outcome-blind). It answers REVIEW R-M2 on the
orchestrator's ruling of 2026-11-05. **It moves no bar, fold, deal, seed, z, β, learning rate, epoch budget or class
rule, and it changes no class.** It changes what a GO licenses, and it makes §7's placebo, already scheduled on GO,
the gate for that licence.

- **Why.** The control is matched on deals, engine, opponent and regime, but not on training dose: the student had
  ≈ 1M row-passes of fine-tuning and P\* had none. A GO could therefore come from fine-tuning toward sharp targets
  rather than from S′'s choices. The dose-matched control is §7's PLACEBO student: the same recipe, rows and seeds,
  with Q̂ permuted.
- **The licence, as amended (§1's GO row).** GO keeps its class, its wording and its other licences: registering
  ExIt-0H, and naming the students an X54 candidate on X54's own terms. **Registering ExIt-1 now needs GO AND the
  placebo attribution SEARCH-INFO.**
  - SEARCH-INFO: the student − placebo lower bound (95 %) is above 0.
  - GO with NOT-SEPARATED licenses only the descriptive claim, "a fine-tuned P\* beats P\* against Kakuna on
    unseen teams". ExIt-1 is not funded on it.
- **The schedule (§7, unchanged in substance).** On a GO at look 1 or at look 2, the placebo is the GO path's next
  step, before ExIt-1 is registered:
  - three placebo students, one per fold, trained by B2's recipe on the same rows, seeds and learning-rate grid,
    with each label row's Q̂ permuted among its legal actions by `default_rng([6599, row index])` before π′ is formed;
  - exported by B3 at the play pin;
  - played on look 1's deals (unit seeds 300–344, 20 a seed, on each student's read panel) as lone one-game units at
    the play pin on E4.
- **The attribution statistic.** It is §6's statistic with the placebo's unit in place of P\* raw: D =
  y(student) − y(placebo) per deal, the panel means, their mean, the within-panel unit-seed bootstrap
  (B = 4,000, `default_rng(20261106)`), z = 1.96 for the 95 % bound, worst-case imputation adverse to SEARCH-INFO
  (student 0, placebo 1), and the root-rule-8 boundary (a bound within 0.0005 of 0 is NOT-SEPARATED).
- **Cost.** It is unchanged: §8's ≈ 37 CPU-h on GO already counts the placebo's 2,700 units.
- **Owed before a GO's placebo trains.** B2's `--placebo` permutation and the attribution rule are not built. They
  carry no weight on any class of this read and are built only on GO.

## Amendment 2 (2026-11-05, before any student trained, any student logit or any game): the pin amendment

The pin amendment §9 and the Status owe before the first student trains. It names the code, the bank and the play
pin, and disposes of every REVIEW finding. **It moves no bar, fold, deal, seed, z, β, learning rate, epoch budget or
class rule.**

**The code.** Builds B1–B5 landed at `e0dc4e466..c9d86ef53`:

| build | commit | what it is |
|---|---|---|
| B4 rule, `games` subset, `PENDING` | `fb6cd6c59` | `main.reads.rules_distill_play`; `by_hand.py` |
| B5 | `320a535d4` | the four read specs |
| — | `76db0229a` | the package docstring names the rule |
| B1 | `d174672ea` | `main.x65.bank` |
| B2 | `858dc283c` | `main.x65.student` |
| B3 | `6fcaaf258` | `main.x65.export` |
| B4 meter | `9a7ac9238` | `main.x65.delivery` |
| tests | `03ac60ac7`, `c9d86ef53` | the specs test's home path; the export's slow test |

The review's fixes landed at `9b6c0541d..b074e08d8` (`d6b5ba393`, `f235bb796`, `530d26d0f`, `d9a0cc58f`,
`2f782ed77`, `d7e2d5193`, Amendment 1 `df7288a04`, `b074e08d8`). The students train at the code as of this
amendment's commit; `student.train` refuses any bank but the registered one below.

**The bank** (B1, built outcome-blind from `c9d86ef53` on E4, `~/gen3ai_archive/x65_exit0_2026-11-05/bank/`):

- **Manifest sha256 `d13fd55f10ba649ad8e728ce1585e4203c4200148d772919eb175d3273992b7a`** (`bank_manifest.json`),
  built at `c9d86ef53` (clean) on E4 from S′-LIVE's `record.json` (sha256 `1975d3d3…1bc2`), with the anchor rows'
  greedy check run on P\*'s bytes (`814823c5…91ed`). `python -m main.x65.bank check --manifest-sha256 <it>` passes.
  This is the bank the students train on; `student.train --bank-sha256` refuses any other.
- 270 shards; 5,398 counted games (S′-LIVE's 2,700 deals × 2 arms, less its two uncounted units); **268,905 decision
  rows**, of which **97,395 are label rows**. Wall time 6,351 s at 5 workers, nice 19 beside a live run.

| panel | label rows recorded | kept (verified) | anchor rows (raw / searched arm) | searched-arm digests checked / matched |
|---|---|---|---|---|
| H | 32,020 | 32,020 | 42,249 / 42,035 | 32,434 / 32,434 |
| F | 33,192 | 33,192 | 49,974 / 48,154 | 38,318 / 38,318 |
| p72 | 32,183 | 32,183 | 44,132 / 42,361 | 32,642 / 32,642 |

- **Every label row is verified: 97,395 / 97,395 (100 %), drop share 0 on every panel, so the bank is not VOID.**
  The digests checked include the excluded decisions' (the anchor rows' path check): 103,394 / 103,394.
- Every recorded `at`, `update`, `log_sha256` and re-score `log` was reproduced (0 mismatches).
- 0 games failed to place; 0 anchor rows failed the arrival-order cross-check; 0 rows failed P\*'s greedy check.
  25 rows were excluded from that check as near-ties (two logits within 1e-4) and kept.
- F holds 206 fewer label rows than `bank_count.json`'s 33,398: that count read every final dir, and
  `pstar_s__forum72__337__g19` (exhausted, uncounted) is not in the counted set.
- One label row (H) has a π′ mode that is not the recorded pick, a near-tie broken by S′'s policy-order tie rule.
  It is reported and kept: its target is still the registered π′.

**The play pin is `320a535d4`** (REVIEW R-m7), the landed B5 commit.
- Its play path is byte-identical to S′-LIVE's `0c76b521e`: `main.reads.playpath.compare` finds 0 differences over
  1,732 files (digest `ff2a4b4484ba`). Later commits differ from it only in `src/main/reads/__init__.py`'s docstring.
- It has no Rust diff from `0c76b521e`, and it stamps E4 (`port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`) once
  the checkout carries main's untracked `src/rust_env/Cargo.lock`. That file is part of the env stamp. The playing
  checkout is built `--with-rust`.
- Its `main.reads` plans the four specs. The operator verbs (`status`, `decide`, `record`) run from the newer
  checkout, whose `distill_play` carries R-m5's pairing check.
- **B3's export check passes there.** P\*'s own weights, round-tripped through `export.write_zip`, give bit-identical
  logits at this checkout and at a detached worktree at `320a535d4`, through the bare loader (`load_policy(...,
  "bare")`). Both the trainer's function and the play path's masked logits match on the 64-row fixture (sha256
  `bb4cf83d…`). Each student's export runs `python -m main.x65.export write --pin-root <a worktree at 320a535d4>`,
  and `export.PLAY_PIN` is now `320a535d4`.

**What the specs still carry as `PENDING`**, filled only once the students are exported (a spec with a pin is refused
by `check-spec --pin-pending`, so the pin and the student shas land together):
- each student's zip path and sha256;
- `pin: 320a535d4`;
- the import hashes of the earlier looks' final specs, in the order the B5 report gave.
- The bank sha256 below is written into the four specs now.

**The REVIEW's findings, disposed.**

| finding | disposition |
|---|---|
| R-M1 the delivery scorer untested | **FIXED** (`530d26d0f`): `score()` runs under test with its forward stubbed, row by row; `delivery_by_hand.py` recomputes Δagree, the pooled Δ, the LB and DELIVERED from the score files with its own loops. Mutation M15 KILLED |
| R-M2 GO licenses ExIt-1 before the placebo | **FIXED** by Amendment 1 (`df7288a04`) |
| R-M3 three bank guards untested | **FIXED** (`d6b5ba393`): mutations M4, M5, M25 KILLED |
| R-m1 B1's named test missed the identity guard | **FIXED** (`d6b5ba393`). The probe's miss is removed by `rebuild_row`'s decision-index check (the naive prefix opens decision 22, not 23); the test asserts that reason. A one-byte-wrong rebuilt row now reaches the `frame_sha256` guard through the real path |
| R-m2 two trainer guards untested | **FIXED** (`f235bb796`): mutations M9, M26 KILLED |
| R-m3 "both cannot hold" | **CORRECTED.** §1's sentence is false. GO and NULL both hold when 0 < LB_go and UB_null < +2 pp, that is when SE < 2 / (z_go + 1.96) pp. The rule checks GO first (tested), so the class is GO. The record says "NULL's test also held" whenever both hold; the rule's `tests` block carries both |
| R-m4 the §5 leak split degenerate | **FIXED** (`d9a0cc58f`). The leak block's `residual` / `partial` are the preset's static leak names, the same on every record. §5's reported split is now `attribution_miss` ≥ 1, `rewritten_moves` ≥ 1, and neither |
| R-m5 no per-pair START check | **FIXED** (`2f782ed77`). A counted pair whose captures' START documents differ in the format, the seed, either packed team or our core side, or cannot be read, is a named VOID (`pairing`), in the rule and in `by_hand.py` |
| R-m6 G-BANK only proves the time it runs | **ALREADY SO, now pinned** (`d7e2d5193`). The queue runs seeds outermost, then groups in spec order. G-BANK is first, so each seed plays its 3 replays and then its 60 student games: the 135 replays are spread through the look |
| R-m7 the play pin | **NAMED** above: `320a535d4`, with the export check run there; `export.PLAY_PIN` follows (`b074e08d8`) |
| R-m8 sets overlap | **ACCEPTED, wording made precise.** No two panels share a team or a six-species core. Exact sets (species, item and moves) are shared: H–F 51, H–p72 63, F–p72 44, of 343 / 255 / 373. Every "unseen teams" in §1 means teams, not sets. No class's licence claims unseen sets |
| R-m9 the drop budget | **NOTED, and §4 corrected.** The cap is 0.5 % of 900 deals a panel, so 4 drops pass and the 5th VOIDs. §4 (line 129) says look 1 carries "two banked raw drops"; it carries ONE, `pstar__heldout72_hpfix__337__g03` on H. S′-LIVE's other OP-7 drop is its searched arm's, whose raw unit counted. So H can absorb 3 more drops at look 1, F and p72 4 each |
| R-m10 four more bank guards untested | **FIXED** (`d6b5ba393`): mutations M7, M20, M21, M24 KILLED |
| R-m11 seeds 600–674 | **ACCEPTED.** They are disjoint from every registered spec and played unit. FINDING X65-F3 has no guard: S′-HONEST's registration must avoid them. This amendment does not touch S′-HONEST's files |
| R-m12 two hazards | (a) the bank finished inside its 5-h timeout (6,351 s); (b) **FIXED** (`d6b5ba393`): the bank-build test pins the engine check both ways instead of depending on the untracked `Cargo.lock` |

**Readings accepted.**
- The review accepted the builders' readings (the design doc's §7).
- The B4 / B5 builder's readings also stand:
  - SE is the sample SD with divisor B − 1;
  - a fresh `default_rng(20261106)` per estimate;
  - the look-2 drop cap is over the look's cumulative deals;
  - a look 1 that is neither GO nor NULL is named CONTINUE;
  - the delivery file's key set is exact;
  - G-BANK is re-checked at look 2 from the imported replays;
  - NOT-DELIVERED is decided with `main.reads decide` on a planned, unplayed state, its record being `decision.json`
    plus the delivery file.

## Amendment 3 (2026-11-06, after the students trained, before any read-panel logit or any game): the specs filled

Written after the three students were trained and exported (the GPU session, 11-06 00:37), before the delivery gate
scored any read-panel row and before any game of this read. It fills the specs' `PENDING` values that Amendment 2
lists, and nothing else. **It moves no bar, fold, deal, seed, z, β, learning rate, epoch budget or class rule.**

**The students** (`~/gen3ai_archive/x65_exit0_2026-11-05/students/<fold>/`; each `pick.json` is §3's blind pick on
inner-validation L, each `model/export.json` is B3's export at `320a535d4`, `bit_identical: true` on the fixture
`bb4cf83d…`, loader `bare`):

| student | read panel | pick: lr, checkpoint, step | inner-val L (label KL, anchor KL) | zip sha256 |
|---|---|---|---|---|
| S_H | H | 1e-5, 5, 344 | 0.7011 (0.6836, 0.0175) | `7af676dd0be5f536494de45fab4a4a9a7da02f3c2e1b3058bbfb8a46685f3f44` |
| S_F | F | 1e-5, 2, 169 | 0.6994 (0.6877, 0.0117) | `e400efb5819d4a77b7c9fc97710832e8139f2fca27bdb32d3d9794baad407f17` |
| S_P | p72 | 1e-5, 12, 744 | 0.7228 (0.7010, 0.0218) | `ec767f51c793df797a451a44b4f75df2a017303e62e1ca1067c25fde005e1ac3` |

- The GPU session's FINDING, recorded before the gate: inner-validation L moved only ≈ 2 % from P\*'s, and the picks'
  anchor KL is 0.012–0.022. The delivery gate (§5) exists for exactly this, and its verdict stands as registered.

**The specs.** Each spec now carries `pin: 320a535d4` and the three students' zip paths and sha256s; each later
look's import of an earlier look names that look's committed spec, filled in the order look 1 → look 1 fresh and
look 2 → look 2 fresh. `check-spec` exits 0 on all four, with the one play-path difference at HEAD Amendment 2 names
(`src/main/reads/__init__.py`'s docstring).

| spec | sha256 |
|---|---|
| `read_spec_look1.json` | `1d4b6d40e430d9f81473aa671b81c4e1381bd8d592d3fe2e1fcc8a8f10b95fcd` |
| `read_spec_look1_fresh.json` | `aff9351771526733b71e2571373d6b794adca9fa41b762fce505fb74fca15ad7` |
| `read_spec_look2.json` | `596fd53f35071d3425091d9edcc1030b3112ee1ab35ef4b67730fa179721ad5c` |
| `read_spec_look2_fresh.json` | `960d33773fdc1f4270f9193935064362bfa470a151bd494a41d621bc6aa44ef9` |

`src/main/reads/x65_specs_test.py` pins the filled values and each import against the committed spec's bytes.
