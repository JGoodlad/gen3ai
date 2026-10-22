# R0, the EXPLOITABILITY CURVE: how much a best responder finds against T\* and P\*, and what one-ply search does to it

**Registered 2026-10-20, BEFORE any R0 run exists and before any game of it is played.** No X35 run's telemetry
(TensorBoard, eval rows, logs) was opened for it. Nothing below may be edited after an R0 read: a correction is a new
dated section at the end. The orchestrator's placement ruling is the launch sheet's §0 **AM-R0**
([`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md)). The kit arms and the read are `src/main/x35/r0.py`
(`toy_r0`, `r0`); the argvs' `checkargs` transcripts are in [`r0/`](r0/).

**Why (the owner, 2026-10-20):** "at some point we should train an exploiter either toy or prod against prod to see how
much head room we have etc. and then exploiter + pi search just to know." R0 also closes ruling C4's deferral (F27 /
F-LS-19: "R0 has no registered argv"), supplies LG-1's R0 (A6's trigger, the R2 baseline) and the X26 collapse
battery's B7 ("a fresh best responder's gap at P\*", scoped in `3e8a4124`).

Words: **BR** = best responder, a fork of a frozen target trained against that target ALONE. **gap** = P(BR beats the
target) − 0.5 (`main.best_response_gap`'s definition; ties are not wins). **pp** = percentage points.

## 1. The two runs

| | **R0-toy** (`toy_r0`, run `rb_x35t_r0_s3631`) | **R0-prod** (`r0`, run `rb_x35_r0_s3531`) |
|---|---|---|
| target = fork parent | T\* = X34 `checkpoint_290000087_steps.zip` (sha `cfa7f7be…`) | P\* = X26 `checkpoint_280000170_steps.zip` (sha `814823c5…`) |
| pin | the TOY pin `4609c155` (`x35_pin_batch1_toy.txt`) | the PRODUCTION pin, which waits for FP32_ALT like every production arm (AM-8) |
| length (the stop rule) | `--steps` 310,000,087 = T\* + 20M | `--steps` 300,000,170 = P\* + 20M |
| seed | 3631 | 3531 (the G-E readers' seed, so R0 and `ge_c0` share the dice) |
| GPU-h | **≈ 1.4** (20M at X34's measured 4,719 steps/s = 1.18 h, + 0.2 start) | **≈ 4.4** (20M at X26's measured 1,333 steps/s = 4.17 h, + 0.2) |

**The argv** is the registered G-E reader's (launch sheet §4.4; F-LS-19: "R0 is the SAME argv with P\* as the
target") with +20M instead of +5M, and the toy side's two extras:

```bash
--model S --exploiter S --no-self-play --trainee-teams <the pinned offense 5-team archetype> \
  --fork-lr 5.6e-05 --fork-lr-freeze --behaviour-check warn --steps <S + 20000000> --checkpoint-every-steps 1000000 \
  [toy: --snapshot-ladder-games 0 --supply-starve-cycles self_play_pool=0] --device cuda --seed <seed> --run-name <run>
```

- One BR per side, piloting the offense archetype, as every exploiter of record did. The target draws its teams from
  the pool. The dose is frozen at production's `recipe.fork` rate, so it is constant over the run.
- **Production R0's first 5M IS LG-1's registered R0.** `r0.prefix_cell` pools its own in-run vs-target rows at or
  below P\* + 5M: that is A6's trigger (gap ≥ 0.15) and the R2 baseline. LG-1's separate R0 (1.1 GPU-h) is therefore
  absorbed.
- **Validated in the kit commit that follows this registration (CPU, rule 5b, `CUDA_VISIBLE_DEVICES=""`):** `checkargs` exit 0, 0 unrecognized, `✓ fork arch`,
  `[ForkLR] ✓ this fork names its own dose`, `self_play False (TYPED)`:
  - toy at the toy pin `4609c155` (14 flags);
  - production against P\*'s pinned parser `@9c033489` and at main HEAD `cd8690c8` (12 flags each).
  - `r0_test.py` pins the kit's argv to these transcripts. The launcher `--dry-run` was NOT run (F-R0-3).
- **Stop rule.** Each run ends at its fixed `--steps`, with no stop on a score. It ends early only on a `FATAL_*`, a
  4th crash, or a K9(b) ceiling read by hand (`warn`, X26's / X34's ceilings, as every X35 arm). At the end the
  Training session runs `kit k9b` (AM-6; R0 is side-pinned, so pre-check 11 covers it).
- **Extension rule.** A curve that reads RISING at the cap (§2) licenses NOTHING by itself. The orchestrator MAY register
  a resume to 2 × 20M (same run, same pin), as a new dated section here.
- **Retention.** No R0 checkpoint is thinned before its curve is read: `r0.curve_plan` refuses a missing read point.

## 2. The read: the exploitability curve (CPU, after each run ends)

`python -m main.x35.r0 plan --run-dir models/<run> --call <side R>` lists the files. Then
`python -m main.x35.r0 play --run-dir models/<run> --call <side R> --out r0/curve_<run>.json` plays them and writes the
record once. It runs under rule 5b and plays on the Rust eval core.

- **Read points** k = 0, 2M, 4M, …, 20M (11). Each is the first checkpoint in [fork + k, fork + k + 1M), with
  `final_model.zip` at the cap. **k = 0 is the start file itself**: the BR's weights at the fork ARE the target's.
- **Each point plays 400 games, greedy vs greedy**, 80 per offense team. The BR pilots its pin; the target draws from
  the pool. This is the regime of the run's own vs-target series and of `best_response_gap --play --greedy`.
  - The same seed is used at every point, so every point plays the same deals.
  - The final also gets a SECONDARY read in the training regime (stochastic@1, `--stochastic`), reported, never classed.
  - Cost: ≈ 4,800 games a side, ≈ 1.2 CPU-h a side, from G1's "≈ 9 h of CPU for 36k games" (UNVERIFIED for the toy).
- **gap(0) is NOT zero by construction.** It is the offense pin's matchup against the pool, played by the target on
  both sides. The curve therefore reports gap(k) AND the LEARNED part gap(k) − gap(0).
- **The class** (`r0.classify`, deterministic; rule 8 at 0.0005):
  - **H, the headroom** = the pooled gap over 16 / 18 / 20M (1,200 games). Its learned part is H − gap(0), with a
    Newcombe 95 % interval.
  - D = pooled(16–20M) − pooled(10–14M), Newcombe 95 %.
  - **RISING** iff lower(D) > 0. Otherwise **SATURATED** iff upper(D) < +5 pp. Otherwise **UNRESOLVED**.
  - **t90** = the first k ≥ 2M whose gap exceeds gap(0) + 0.9 (H − gap(0)). It is a point read, noisy, and descriptive.
    **NO RISE** iff H − gap(0) < 2 pp.
  - Any point with > 25 % timeouts makes the curve **INCONCLUSIVE**.
- **Toy vs production** is DESCRIPTIVE: H_toy − H_prod with a Newcombe interval. The two differ in architecture, in the
  target's strength (P\* beats T\* 0.59, C5) and in the BR's relative dose (F-R0-5).

## 3. Predictions (registered first)

**The anchor, and where it came from.** X55's P4 registered R0's gap(P\*) as **+14 pp, 80 % [+6, +22]**, with
P(gap ≥ 0.15) ≈ 0.45 ([`PREDICTIONS.md`](../x55_league_headroom_2026-10-20/PREDICTIONS.md)). Its basis, "the old lineage
read +14.0, +15.75 and +20.0 on its plateau parents", traces to PRE-ERA (`ai_v13`) offense exploiters of the plateau
parent `ai_v13_12_plateau` @95M:
- **+14.00**: arm A, pooled over 4 cycles = 8M (population loop round 1);
- **+15.75**: `ai_v13_13_exploit5_offense`, pooled; its ENDPOINT 0.700 is the **+20.0**;
- A's seed replicate A2 read **+12.00** [+7.15, +16.62] (ledger L21338).

So the anchor is an 8M pooled statistic of a different lineage, quoted for a 5M reader (F-R0-1). The old readers' curves
were FLAT by 8M at a matched dose (A, A2 and RC flat; RB still rising, 0.52 → 0.60; ledger L21338). This is the
saturation prior below. X55's P4 stands unchanged and is scored as registered.

**The curve model behind the numbers (a GUESS):** gap(k) = gap(0) + L (1 − e^(−k/τ)).
- Production: L = +18 pp, τ = 3.5M, so t90 ≈ 8M.
- Toy: L = +22 pp, τ = 2.6M, so t90 ≈ 6M.
- The toy is a smaller network with no belief tokens and is 9 pp weaker than P\* head to head, so it should be easier to
  exploit. But its BR shares that capacity, so the difference is predicted SMALL.

**P-R0-1. The gap at each read point** (pp; point, 80 % interval). gap(0): **0, [−6, +6]** on both sides.

| k (M) | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **production** | 8 [0, 16] | 12 [3, 21] | 15 [5, 25] | 16 [6, 27] | 17 [7, 28] | 17 [7, 29] | 18 [7, 29] | 18 [8, 30] | 18 [8, 30] | 18 [8, 30] |
| **toy** | 12 [2, 22] | 17 [6, 28] | 20 [8, 31] | 21 [10, 33] | 22 [10, 34] | 22 [10, 34] | 22 [10, 34] | 22 [10, 34] | 22 [10, 34] | 22 [10, 34] |

- **Headroom H:** production **+18 pp, 80 % [+8, +30]**; toy **+22 pp, 80 % [+10, +34]**. The learned part equals H to
  within gap(0)'s interval.
- **LG-1's R0 statistic** (production, in-run pooled to P\* + 5M, 2 cycles): **+10 pp, 80 % [+3, +18]**. That is below
  X55's +14, because it pools the 2M cycle. P(≥ 0.15, A6's trigger) ≈ 0.25 here, against X55's 0.45.

**P-R0-2. Toy vs production.** H_toy − H_prod = **+4 pp, 80 % [−8, +14]**; P(H_toy > H_prod) ≈ 0.65.

**P-R0-3. How fast it saturates.**

| | t90 | P(SATURATED) | P(UNRESOLVED) | P(RISING) | P(NO RISE) |
|---|---|---|---|---|---|
| production | 8M, 80 % [4, 14] | 0.55 | 0.30 | 0.15 | 0.05 |
| toy | 6M, 80 % [2, 12] | 0.60 | 0.30 | 0.10 | 0.05 |

P(t90_toy ≤ t90_prod) ≈ 0.6.

**Scoring.**
- Each point is scored against its 80 % band. About 9 of 11 inside is calibrated; ≤ 6 is a miss of the curve model.
- H is scored against its interval, and each class gets a log score.
- X55's warning applies: its point predictions read about 2× too confident (REPLAN §5). These bands are wider.
- A miss is a FINDING about the headroom model, never a reason to re-read.

## 4. "Exploiter + pi search, just to know" (DESCRIPTIVE, CPU, after each R0)

**The search.** L1's anchored one-ply search, as L1 registered and fitted it
([`search_l1_l5_2026-10-20/`](../search_l1_l5_2026-10-20/PREREG.md); `search_ledger.md` N5 / Q13):
- the 8 fully honest worlds F (Smogon-usage draws kept only if they replay what the searched side saw; `dice="seed"`);
- the TRUNC-4-K8 leaf: 8 playouts, 4 turns, then the critic;
- **piKL at the fitted λ = 0.015**: pick argmax Q(a) + λ·log π(a). This is the one cell L1 read PAYS-SCOPED.
- It searches every decision of the searched side.

| cell | searched side | opponent (raw, greedy) | the number |
|---|---|---|---|
| **S-BR** (what search adds to the exploiter) | the BR's final + piKL | its target | Δ_S = WR(BR+search) − WR(BR raw) |
| **S-T** (the MIRROR: does search defend?) | the TARGET + piKL | the BR's final | Δ_M = WR(target+search) − WR(target raw) vs the BR |

- **The deals.** Each cell plays **100 deals**: offense pin × pool draw × battle seed. The raw arm plays the SAME 100
  deals (common random numbers), and the difference is a paired bootstrap over deals.
- **No mirrored pairs.** They are refused for a pinned team (P13). Team variance therefore sits equally in both arms.
- **Also reported:** the DEVIATION RATE, the share of searched decisions where piKL's pick is not the policy's greedy
  choice.
- **Order.** Toy cells after R0-toy's curve; production cells after R0-prod's. Four cells in all.
- **Cost (UNVERIFIED).** ≈ 32 searched decisions per game (the median game is 29–40 turns) × ≈ 60 CPU-s per decision
  (≈ 28 s at 4 worlds on a contended box; 8 worlds 42–96 s) ≈ **32 CPU-min a game**, so ≈ **53 CPU-h a cell**. Four
  cells ≈ **213 CPU-h**: ≈ 3 days on the 3 agent cores at nice 19, so it crawls.
  - **Re-size rule** (cost is not an outcome): if the first 10 games of the first cell measure more than 2× the
    per-decision estimate, every cell drops to 50 deals. This is fixed before any outcome is read.
- **Predictions (GUESSES, registered):**
  - **Δ_S = +1 pp, 80 % [−6, +8].** L1 found the anchored search ≈ the policy on held-out banks. Its rollouts also play
    the BR on both sides, so it models the target as the BR (F-R0-9).
  - **Δ_M = +3 pp, 80 % [−4, +10]** for the target. The BR exploits the target's DETERMINISTIC greedy replies, and any
    deviation breaks some of those lines even when it is no better in general. That is robustness by unpredictability,
    not by better play.
  - P(Δ_M > Δ_S) ≈ 0.6. Deviation rate ≈ 10–20 % (UNVERIFIED).
- **The honest-world caveat.** The search never sees the opponent's true sets. The honest leaf holds little the policy
  lacks: L1 found the acting rule does not bind, the sampler does. A null here says "honest one-ply search adds
  nothing", never "search cannot exploit or defend".
- **Not a verdict.** At n = 100 the paired interval is ≈ ±8–10 pp (UNVERIFIED). These cells decide nothing and are
  quoted only with that caveat.
- **Not ready to run: the harness is a BUILD** (F-R0-8). The search module (`main.value_search.search_decision`) searches
  a decision of a RECORDED log. No live player that searches inside a game exists; S0b's was never built.
  - L1's decision-level metric (IMP with the true-world referee) cannot stand in for it. Its referee plays the searched
    network on BOTH sides, so it never faces the BR (for S-T) or the target (for S-BR).
  - The live player is ≈ 1 agent-day (UNVERIFIED) and out of this unit's scope (rule 9). It is the item to queue before
    the first cell.

## 5. GPU-h added, and the order (AM-R0 rules it)

- R0-toy ≈ 1.4 GPU-h, immediately after the GPU battery and before T1b, at the toy pin.
- R0-prod ≈ 4.4 GPU-h, right after P1's C0 seeds (s3501–s3503) and before A1, at the production pin.
- **Total ≈ 5.8 GPU-h.** LG-1's own R0 (1.1) is absorbed, so the plan grows by **≈ 4.7**.
- CPU: the curves ≈ 2.5 CPU-h; the search cells ≈ 213 CPU-h, after the live player is built.

## FINDINGS (standing rule 7)

- **F-R0-1.** X55's +14 pp anchor mixes two pre-era runs' statistics and a different budget (+14.00 and +15.75 pooled
  over 8M, +20.0 an endpoint). The seed replicate read +12.00. This registration's 5M-prefix prediction (+10) is below
  X55's; X55 is not edited.
- **F-R0-2.** The GPU-h are UNVERIFIED. No Rustboro-era exploiter has run (F-LS-21); an exploiter's step rate is not a
  self-play run's, and production's rate under FP32_ALT is unmeasured (F-LS-11). Read the first 1M of each run.
- **F-R0-3.** The launcher `--dry-run` was NOT run. Without a lease the GPU is off-limits, and the launcher's
  CUDA-visibility preflight refuses a `--device cuda` dry run under an empty `CUDA_VISIBLE_DEVICES`. `launch_x35.sh`
  runs `checkargs` and the dry run at GO, on the real pin.
- **F-R0-4.** The production pin does not exist yet (FP32_ALT). The argv was checked against P\*'s pinned parser and
  main HEAD only.
- **F-R0-5.** The BR's dose is 5.6e-5 on both sides: 0.46× P\*'s saved rate (1.2056e-4) but 0.39× T\*'s (1.4468e-4).
  The toy BR therefore learns at ≈ 0.84× production's relative step, a confound of P-R0-2 and P-R0-3.
- **F-R0-6.** R0 measures exploitability ON THE OFFENSE ARCHETYPE (a subgame restriction), as every exploiter of
  record did. gap(0) carries the team matchup. A full-pool BR would measure policy exploitability without it, but has no
  meter (`--play` needs a pin). It is named, not built.
- **F-R0-7.** The 11 read points share their deals (one seed). D's Newcombe interval assumes independence, so it is
  conservative: it makes SATURATED harder to read, never easier.
- **F-R0-8.** The search cells need a live one-ply search player, which is not built (§4).
- **F-R0-9.** Both search cells play rollouts with the searched network on both sides (`playout.py`), so neither models
  the opponent it faces. An opponent-aware leaf (the real opponent's network on the other side) is the natural second
  arm. It is named, not designed.
- **F-R0-10.** Whether the search harness at HEAD loads a toy checkpoint (no belief tokens, the toy architecture) is
  UNVERIFIED. So is whether `r0 play` at HEAD loads the pinned R0 checkpoints. If not, run the read in a worktree at the
  side's pin.
- **F-R0-11.** `kit disk` does not count R0 (F-LS-22's rule). R0 writes 20 checkpoints a run, ≈ 0.44 GB on the toy and
  ≈ 1.1 GB on production (from T\*'s 22 MB and P\*'s 55 MB).
- **F-R0-12.** R0 is SIDE-PINNED, so its K9(b) at the end joins pre-check 11. An R0 VOID HOLDs the next GO on either
  side, as F-LS-16 reads AM-6.
- **F-R0-13.** The placement is not a gate (as F-RP-6). `stage-check` gives `toy_r0` ALLOWED at any time and `r0`
  ALLOWED once FP32_ALT is named.
- **F-R0-14.** `r0 play` (11 checkpoints against their target through `untaught_meter.play_cells`) has not run end to
  end; no R0 run exists. Its tests cover the plan, the class and the refusals. The first read runs `plan` first.

## 2026-10-21: CORRECTION to §2's read-point rule (the first R0-toy read, before any game was played)

§2's "the first checkpoint in [fork + k, fork + k + 1M)" cannot be met by the trainer. It saves at the first call past
each ABSOLUTE multiple of `--checkpoint-every-steps` (`main.train.constants.checkpoint_due`), so the +6M save of the
fork at 290,000,087 is `checkpoint_296000066`, 21 steps SHORT of fork + 6M, and `r0 plan` refused the run. The window is
now anchored on the fork's boundary B = the fork step floored to a multiple of 1M: [B + k, B + k + 1M). For R0-toy it
picks the save of each nominal 2M point (292,000,095 … 310,000,106; `r0_test.py` pins this run's real grid). The fix
was made from the file names alone, before any game was played. Also new: `r0 play --ledger DIR --max-points N` banks
each point and resumes in chunks (the deals are identical: the meter's seed depends only on (seed, team index)).

**R0-toy is READ** (`r0/curve_rb_x35t_r0_s3631.json`; the ledger entry of 2026-10-21): RISING, H +11.8 pp.

## 2026-10-22: R0-prod is READ (no correction)

**R0-prod is READ** (`r0/curve_rb_x35_r0_s3531.json`; the ledger entry of 2026-10-22) as registered, with the
2026-10-21 window rule and no new correction: SATURATED, H +11.1 pp, t90 8M, LG-1's 5M statistic +6.0 pp.
