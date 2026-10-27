# STATUS — where gen3ai stands, on one page

**As of 2026-10-26.** This page is the dashboard: what we are trying to do, where each goal stands, and what is
running. It gives each number, its interval and the commit that proves it. The reasoning behind these numbers is in
[`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md), the plans are in
[`endstate/README.md`](endstate/README.md), and the research ledger wins any disagreement. A backticked hex string
is a commit on `main` (`git show <sha>`).

**How to keep it current:** a new verdict, a launch or a finished job rewrites its row here in the same commit.
`src/status_doc_gate_test.py` fails when a cited commit is not in `main`'s history, or when the date in the
"Running now" heading is more than 3 days older than the newest commit. This page replaces the orchestrator's
status digests in chat. It does not replace the status block each design doc keeps for itself.

**Terms used below.** **P\*** is our strongest checkpoint: the X26 run at 280M steps, frozen as the reference at
`d172c10ed`; the named baseline `pstar_plateau`. **Kakuna** is Metamon's strongest public model (142.8M parameters;
ours has 2.61M, 2.03M of them in the feature extractor: `ARCHITECTURE.md`). **home72** is our 72 training teams.
**Unfamiliar teams** (pooled as "U" or "H") are teams our model never trained on. "pp" means percentage points of win
rate. **NOT DETECTED / UNDECIDED** means the interval spans zero, so we cannot tell the lever from noise.

---

## 1. The goals, in the owner's order

| # | goal | status | best number [95 % interval] | proof |
|---|---|---|---|---|
| 1 | **The strongest baseline** | **AT ITS PLATEAU.** P\* = X26 @ 280M. Training on to 480M changed nothing. | P\* vs the production baseline: 0.694 [0.666, 0.721]. The 480M continuation vs P\* against Kakuna: +0.08 pp [−3.90, +4.06], FLAT. | `d172c10ed`, `204204bbf` |
| 2a | **Beat Metamon on our teams** | **MET.** | P\* vs Kakuna on home72: **0.598 [0.549, 0.644]** | `3010c81ba` |
| 2b | **Beat Metamon on unfamiliar teams** | **NOT MET**, about 4 pp short. The lower bound must clear 0.50, which needs about 0.53. | P\* 0.487 [0.459, 0.515]. No lever's interval clears 0.50. Highest point estimates, each against its own control and none detected: the belief prior mix (λ = 0.25) 0.502, D1 0.492, D3 0.491 (look 1). | `3010c81ba`, `f299a7243`, `eb99ab362`, `58287c525` |
| 3 | **Know where we are** | **MAPPED.** The deficit is **piloting** an unfamiliar team, not facing one. In-house meters cannot see it; only Kakuna games can. Nothing measurable is lost in the opening; the gap shows from level positions after turn 10 (descriptive: `research_state/measurements/piloting_gap_forensics_2026-10-26/`). Whether it goes with the team or the set is OPEN until rc72b (the per-mon evidence restates the result: `h_mid_value_test_2026-10-26/REVIEW.md` B1). P\*'s own critic does not see it (H-MID: CRITIC-MISCALIBRATED-ON-U), as a self-play critic cannot (REVIEW M1). | Piloting deficit, strength-corrected (X26 @161M): +0.123 [+0.039, +0.206]. P\* is level with Metamon's ensemble on unfamiliar teams, ahead of it on home72, and beats every human-only Metamon model (§4). | `8b43eb44a`, `82858f433`, `82f273eff`, `df35d1bb0` |
| 4 | **AlphaStar-style improvement** (league, PFSP, exploiters) | **NO GAIN YET.** The archive-pool and promotion levers are NULL. An exploiter beats P\* easily, so headroom exists. No league arm has run. | Best responder vs P\*: **+11.1 pp [+8.3, +13.8]** (saturates by 8M steps) | `806e50931`, `4290d652a`, `c456b6a09` |
| 5 | **What search needs** | **PAUSED, and not getting closer.** Every search gain we see is graded by our own model ("self-referee"). Graded by a real bot, the gains shrink (descriptive; the registered read is UNDECIDED). At P\* the one gain that replicates is gone. | L1 cell at P\*: +0.0001 [−0.0097, +0.0101]. At the 25M policy: +0.0193. | `f756019d7`, `015f00e20`, `0d0058408`, `73428c077` |
| 6 | **Which collapse, and counterfactuals** | **READ: a treadmill, not a collapse** (one seed, provisional). The policy keeps changing without improving, the opponent pool goes stale and the learning rate halves. No coverage, rank or dormant-unit collapse. No counterfactual source is built into training. | about 30 % of greedy choices change per 20M steps | `3e8a41249` |

**Kakuna's team pack changed (2026-10-27, `gen3_peer_showdown_pack_v1`, SOP H22).** Every forum72 Kakuna read before it
had Kakuna's `sft431` Gyarados play Frustration at base power 1 (its packer dropped `Happiness: 0`), a small constant
handicap on Kakuna's side (about 1 team in 72). The peer now packs as Showdown does, so new forum72, away and pool
Kakuna reads are their own regime: a different regime id, and an import of banked rows refuses.

## 2. Running now (as of 2026-10-26)

| where | job | what decision it drives |
|---|---|---|
| **GPU** (lease: Training session) | **LR-F** s3502 (GO 2026-10-27 07:24, 1,470 steps/s), then s3503; the card frees at about 16:00 PDT 10-27. X59-S's students finished on 10-26; its reads are CPU. | Whether the plateau comes from the learning-rate controller, not from capacity or data. |
| GPU queue, next | **A DRAFT 72-h order, the orchestrator's call** ([`ops/gpu_plan_2026-10-27.md`](ops/gpu_plan_2026-10-27.md)): LR-F s3501 (look 1 is registered at three seeds; the chain runs two) → X48 (the trainer is ready; the kit arm and PREREG are not) → N4 (waits on T69) → X62 (waits on rc72b and its build). | Capacity at the plateau (X48, N4) and team recombination for goal (b) (X62). |
| **CPU** (Kakuna lanes) | **N1-P look 1** (X61P): the three 12,000-procedural-team arms (finished training 2026-10-26), read against Kakuna and in-house. | **GO** funds **X53-N1**, a fresh production run on 10⁴ teams, the survey's top-ranked lever. **NULL** rules out only "N1 at the plateau". |
| CPU, queued behind the H-CTX rc72b read | **D3 look 2**, the final read: all five D3 seeds (s3504 and s3505 trained, `a9c09711d`, `28711af06`) against C0's banked rows. Every game is played on ONE Rust engine, E1 (the arms' training engine), at a dedicated play pin: E1's game code under the newest read harness that predates the Kakuna pack change (PREREG AM-D3-L2). It starts when H-CTX's Kakuna lane ends and shares the heavy slots with N1-P look 1. About 16 h of Kakuna lane and 7.5 h of in-house S. | Whether **set mutation** is adopted as a goal-(b) lever: 2,876 pool teams with 1–2 sets redrawn from Smogon. |

## 3. Results of the last week (2026-10-19 → 10-26)

| lever / read | verdict | effect [95 % interval] | record |
|---|---|---|---|
| A1: archive opponent pool | NULL, final | R1 +0.19 pp [−1.07, +1.46] | `806e50931` |
| SPRT: promote snapshots by a sequential test | NULL, final | R1 −0.25 pp [−1.65, +1.16] | `4290d652a` |
| X47: toy depth 2 → 4 | UNDECIDED, Δ̂ ≤ 0, ended | −0.88 pp [−3.20, +1.44] | `d6e57f2e9` |
| X52: practise 36 teams for 5M (ceiling of practice) | UNDECIDED | practised +3.2 pp [−0.2, +6.8]; unpractised +1.3 [−3.5, +6.1] | `42bb0858d` |
| D1: widen the team pool | UNDECIDED, final | Kakuna, unfamiliar teams: +2.10 pp [−0.92, +5.11]; in-pool −1.61 [−2.90, −0.32] | `eb99ab362` |
| X26c: train on 380M → 480M | FLAT; goal (b) not met | +0.08 pp [−3.90, +4.06] | `204204bbf` |
| D3 look 1: set mutation | CONTINUE (look 2 decides). Read ACROSS two engines: D3 on E2, banked C0 on E1. The guard's 140 / 140 identical games bound the divergence below about 2.1 % of games | Kakuna, unfamiliar teams: +1.27 pp [−2.63, +5.16]; in-pool −0.24 [−1.10, +0.62], non-inferior | `58287c525` |
| Belief prior mix, λ = 0.25 | NOT DETECTED | +1.50 pp [−2.74, +5.74] | `f299a7243` |
| Belief prior mix, λ = 1 (prior only) | NOT DETECTED; H3 not killed | −1.50 pp [−5.47, +2.47] | `71c76dfbb` |
| Human-only Metamon models (FR-1, FR-3) | P\* BEATS all three | SmallRL 0.666 [0.639, 0.692]; LargeRL 0.631 [0.603, 0.658]; LargeIL 0.790 [0.766, 0.812] | `4d8d9eef8`, `df35d1bb0` |
| PastaMittens (Metamon's ensemble) vs Kakuna as the opponent | NOT DETECTED: Kakuna stays the opponent | Δ unfamiliar −0.25 pp [−4.23, +3.73] | `82f273eff` |
| R0: exploitability of P\* | HEADROOM, saturated | best responder +11.1 pp [+8.3, +13.8]; the edge transfers (+14.5 [+7.6, +21.2] vs C0) | `c456b6a09`, `1459416fb` |
| E1-0: a real bot as the search referee | UNDECIDED, no relabel | oracle bot − self −0.024 [−0.054, +0.009] | `015f00e20` |
| Hidden-info maturity, plateau step | GAP-SHRINKS NOT DETECTED; frontier static | G_F change −0.003 [−0.020, +0.015] | `e7048bb1d` |
| L1 replication (piKL, 4-turn leaf) | REPLICATES thinly, pooled over the 25M and P\* anchors; post hoc it is all the 25M anchor's | +0.0097 [+0.0025, +0.0169]; at P\* +0.0001 [−0.0097, +0.0101] | `f756019d7` |
| Piloting-gap forensics (banked Kakuna games, descriptive) | the gap shows after turn 10, from level positions; team vs set NOT separated (the per-mon KO drop is the win gap restated: within won / lost games −0.011 / −0.027; review B1) | level at turn 10: 0.609 home72 vs 0.501 U; seen sets only, per-mon KO edge drop +0.093 [+0.004, +0.176] | `piloting_gap_forensics_2026-10-26/` (ledger 10-26) |
| H-MID: does P\*'s critic see the unfamiliar-team deficit? | **CRITIC-MISCALIBRATED-ON-U**: V's in-game view sees none of it (its 3.5 pp set gap is a turn-1 team prior, mostly P72; review M2); at equal V, P\* wins 8.6 pp less on U; V's ordering (AUC) is equal on both sets. Expected of a self-play critic, so it does not place the memorisation in the critic (review M1) | level at turn 10: Δbias +0.076 [+0.016, +0.139] | `2dd522785`, `h_mid_value_test_2026-10-26/` (+ `REVIEW.md`) |
| N2 identity test: does P\* key on our sets' exact identity? (CPU probe) | COMPOSITIONAL: lever N2 dropped | familiar / unfamiliar policy shift under a set re-draw 1.19× [1.07, 1.33] (bar 1.5×) | `98bcb1146` (PREREG), `n2_identity_test_2026-10-26/` (ledger 10-26) |
| Plasticity probe: has the plateau trunk stopped bending to new data? (CPU probe, X26 25M → 480M) | NO-LOSS: plasticity loss unlikely; the strategy tree takes the capacity branch | refit speed on P\*'s own labels 0.986× X26 @25M's (bar < 1.10); near-dormant units 0.78×; a neutral target refits 1.11× slower than @25M, all of it paid by 175M | `plasticity_check_2026-10-26/` (ledger 10-27) |

## 4. How we compare with the state of the art

| opponent | what it is | our result | record |
|---|---|---|---|
| **Kakuna** | Metamon's best public model; 142.8M parameters, offline RL on human and self-play data | home72 **0.598** (ahead); unfamiliar **0.487** (≈ even, goal not met) | `3010c81ba` |
| **PastaMittens** | Metamon's ensemble | U 0.484 [0.456, 0.512], home72 0.550 [0.501, 0.598]: no stronger than Kakuna | `82f273eff` |
| **Metamon human-only models** | trained only on human replays (SmallRL, LargeRL, LargeIL, 200M) | P\* beats all three (0.63–0.79). Offline RL beats imitation on the same data by about 16 pp. | `4d8d9eef8`, `df35d1bb0` |
| **Foul Play** (search bot; Gen 9 OU champion) | MCTS over a hand-written evaluator | **By proxy only:** Kakuna draws it, 0.490 [0.422, 0.559], and P\* beats Kakuna on home72. **Never read against P\*.** Our old 75M policy lost 0.388 [0.288, 0.497]. | `8cba100f8`, `e00e78da0` |

So: at the state of the art on familiar teams, with a model about 55 times smaller, and level with it on unfamiliar
teams (survey: [`research_state/literature/sota_survey_2026-10-26.md`](research_state/literature/sota_survey_2026-10-26.md),
`fd3a99b65`).

## 5. Decisions waiting on the owner

1. **Cloud GPU (vast.ai).** The plan asks to top up the credit by $15, spend at most $20 in total, upload P\* and its
   pool snapshots (about 170 MB, never human data or credentials), and run a card-gated first workload. Each later
   cloud experiment frees 12–13 local GPU-h for about $5. The question is verbatim in
   [`ops/cloud_gpu_plan.md`](ops/cloud_gpu_plan.md) §6 (`37de8515f`). Recommendation: yes.
2. **The history rewrite.** The plan replaces 61 human account names with salted codes in every past version of
   every file, then replaces `main`'s history on GitHub. Scheduled 2026-10-25, **not done**. The owner's part: the
   go to replace `main`'s history, and a GitHub Support request to purge cached copies of the old commits (only the
   repository owner can file it). Every commit since the names landed gets a new SHA. This page's gate will then
   fail until the cited SHAs are re-mapped from the old-to-new table the rewrite adds.

## 6. The three biggest risks and unknowns

1. **No lever we have measured closes goal (b)'s ≈ 4 pp alone.** D1, D3, X52 and the belief mix each read +1 to
   +3 pp with intervals ±3–4 pp wide. Kakuna games run on CPU at 160–230 an hour on the shared agent cores, so each look costs days. The bet
   left is breadth at 10⁴ teams (N1-P, then [X53-N1](endstate/design_x53_n1_fresh_breadth.md), ≈ 39 GPU-h to a
   one-seed screen). If D3, N1-P and X59-S all miss, the fallback is
   [`endstate/strategy_if_levers_null_2026-10-26.md`](endstate/strategy_if_levers_null_2026-10-26.md), both REVISED
   after their [review](research_state/measurements/strategy_review_2026-10-26/REVIEW.md) (`6fca2c061`). The
   fallback's first branch is now a zero-GPU **precision read**: D1, D3 and N1-P against C0, pooled, on 38,000 new
   Kakuna games at one engine, which sees +1.5 pp with probability 0.76. Its cost is the CPU: about 7–10 days of the
   local Kakuna lane, or about a day on a rented CPU box (an owner question when it runs). X53-N1 now waits on a
   zero-GPU measurement of how far two fresh runs differ: above about 2 pp, no affordable seed count can see a
   realistic +2–3 pp effect.
2. **The meter itself.** It is unverified whether Kakuna's self-play saw near copies of our unfamiliar-team panel
   (assumption A-K, open). Foul Play has never been read against P\*. In-house meters are blind to the deficit, so
   only slow Kakuna reads steer goal (b).
3. **The plateau may be the recipe, not the ceiling.** Plain training is FLAT and the population levers are NULL,
   yet a best responder still finds +11 pp. Search does not pay at P\*. LR-F and N1-P are the next tests of whether
   P\* is the most this architecture can reach.
