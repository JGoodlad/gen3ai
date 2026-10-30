# STATUS — where gen3ai stands, on one page

**As of 2026-10-27.** This page is the dashboard: what we are trying to do, where each goal stands, and what is
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
| 1 | **The strongest baseline** | **AT ITS PLATEAU.** P\* = X26 @ 280M. Training on to 480M changed nothing. Holding the learning rate up (LR-F) did not help in-house: UNDECIDED, final, leaning negative. Per-kind FFN experts (X48) gave no ≥ 2 pp in-house gain: NO-GO, weak (the experts barely moved). | P\* vs the production baseline: 0.694 [0.666, 0.721]. The 480M continuation vs P\* against Kakuna: +0.08 pp [−3.90, +4.06], FLAT. X48 − C0 on S: −0.55 pp [−1.32, +0.21] (90 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/x48_look1.json`). LR-F − C0 on S: −1.13 pp [−2.45, +0.18] (97.5 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/lrfloor_look1.json`). | `d172c10ed`, `204204bbf`, `f7fddc39b` |
| 2a | **Beat Metamon on our teams** | **MET.** | P\* vs Kakuna on home72: **0.598 [0.549, 0.644]** | `3010c81ba` |
| 2b | **Beat Metamon on unfamiliar teams** | **NOT MET**, about 4 pp short. The lower bound must clear 0.50, which needs about 0.53. | P\* 0.487 [0.459, 0.515] (its forum72 third under Kakuna's legacy pack, below). No lever's interval clears 0.50. Highest point estimates, each against its own control and none detected: the belief prior mix (λ = 0.25) 0.502, D1 0.492, D3 0.494 (final, UNDECIDED). N1-P (look 2, final: NULL) 0.500 is M4-hazarded (its corpus overlaps H and P72) and is not read as goal (b); on F it reads +0.55 pp [−1.26, +2.36] over C0. | `3010c81ba`, `f299a7243`, `eb99ab362`, `58287c525` |
| 3 | **Know where we are** | **MAPPED.** The deficit is **piloting** an unfamiliar team, not facing one. In-house meters cannot see it; only Kakuna games can. Nothing measurable is lost in the opening; the gap shows from level positions after turn 10 (descriptive). It does not go with the SETS (H-CTX SUPPORTED): on new sixes of familiar pool sets P\* scores 0.514, below F's 0.540 and 10 pp under home72 on the same engine; the sets keep at most about a fifth of the premium. Why the six matters (memorised sixes, broken cores, no plan) is open (`hctx_rc72_2026-10-26/REVIEW.md` M1). Under a neutral pilot (Kakuna on both sides) rc72b is only about 4 pp weaker, flat across coherence (`hctx_kk_2026-10-27`), so the premium stays mostly team-level. P\*'s own critic does not see it (H-MID: CRITIC-MISCALIBRATED-ON-U), as a self-play critic cannot (REVIEW M1). | Piloting deficit, strength-corrected (X26 @161M, PROVISIONAL, resting on the unverified assumption A-K): +0.123 [+0.039, +0.206], facing +0.017 [−0.062, +0.095]; their difference +0.106 [−0.009, +0.221] is UNDECIDED. P\* is level with Metamon's ensemble on unfamiliar teams, ahead of it on home72, and beats every human-only Metamon model (§5). | `8b43eb44a`, `82858f433`, `c758fcb82`, `905c29f5b`, `f200258f6`, `90708bff5`, `82f273eff`, `df35d1bb0` |
| 4 | **AlphaStar-style improvement** (league, PFSP, exploiters) | **NO GAIN YET.** The archive-pool and promotion levers are NULL. An exploiter beats P\* easily, so headroom exists. No league arm has run. | Best responder vs P\*: **+11.1 pp [+8.3, +13.8]** (saturates by 8M steps) | `806e50931`, `4290d652a`, `c456b6a09` |
| 5 | **What search needs** | **PAUSED, and not getting closer.** Every search gain we see is graded by our own model ("self-referee"). Graded by a real bot, the gains shrink (descriptive; the registered read is UNDECIDED). At P\* the one gain that replicates is gone. | L1 cell at P\*: +0.0001 [−0.0097, +0.0101]. At the 25M policy: +0.0193. | `f756019d7`, `015f00e20`, `0d0058408`, `73428c077` |
| 6 | **Which collapse, and counterfactuals** | **READ: a treadmill, not a collapse** (one seed, provisional). The policy keeps changing without improving, the opponent pool goes stale and the learning rate halves. No coverage, rank or dormant-unit collapse; starvation creeps up +0.03 to +0.06. No counterfactual source is built into training. | about 30 % of greedy choices change per 20M steps | `3e8a41249` |

**Two pack regimes (2026-10-27, SOP H22 / H23).** Kakuna's peer now packs teams as Showdown does
(`gen3_peer_showdown_pack_v1`, `b0441623e`): before it, Kakuna's forum72 Gyarados played Frustration at base power 1
(about 1 team in 72), so new forum72, away and pool Kakuna reads are their own regime and an import of banked rows
refuses. The front end now plays the validator's re-pack, which changes only Foul Play's Hidden Power sets, so Foul
Play reads of the pool are their own regime too (`gen3_frontend_validated_pack_v1`, `951508922`). Foul Play on
`away` cannot run (19 of its 21 team files are refused; `48eeb0b70`); every Foul Play cell so far drew a `home` set.
OUR side has its own split. Before `d8d674b99`, our packer also played Frustration at base power 1. That covers P\*'s,
X26c's and every X35 arm's training (0.13–0.38 % of team draws), and our side's forum72 `sft431` in pre-fix reads.
The arm-vs-C0 bias is ≤ 0.2 pp, and no verdict moves (the ledger's 10-27 pack-regime AUDIT,
[`x35_pack_regime_2026-10-27/`](research_state/measurements/x35_pack_regime_2026-10-27/README.md)).

## 2. Running now (as of 2026-10-27)

Read at 12:05 PDT from the arm chain's status and each read's state directory; the GPU rows at 23:55.

| where | job | what decision it drives |
|---|---|---|
| GPU: X48 DONE 10-28 08:42 | **X48** s3501–s3503 trained (each K9(b) PASS, VOID 4 exact PASS; dose ≈ 1.03× C0's wall). Its read is the CPU row below. | — |
| GPU queue, next (the orchestrator's chains; draft order [`ops/gpu_plan_2026-10-27.md`](ops/gpu_plan_2026-10-27.md)) | **N4** (the toy capacity screen; its model build landed at `c35ffab9e`, its kit arms and pin have not). | Capacity at the plateau, the branch the plasticity probe's NO-LOSS chose (X48, N4). |
| CPU: DONE 10-30 11:04 | **N1-P look 2** (X61P, final), the five forks with half their teams from 12,000 procedural teams: **NULL.** F +0.55 pp [−1.26, +2.36] (N1-P 0.509 vs C0 0.504, 6,000 forum72 games per arm; seeds +1.0 / −0.25 / +1.5 / +1.0 / −0.5), inside ±3 pp; in-house S on home72 −3.28 pp [−4.14, −2.43] (INFERIORITY-NOT-EXCLUDED). Every game on ONE engine (E3) at look 1's commit (AM-N1P-2, `d78700253`); every registered check PASS. Record `x35_breakers_stage_2026-10-17/reads/n1p_look2.json`. | Rules out **"N1 at the plateau"** (share 0.5, 20M) only: not N1 from step 0, the toy screen or a human-derived corpus. X53-N1's trigger (N1-P GAIN) did not fire. X63 unchanged (AM-X63-N1L2: NULL ⇒ unchanged). |
| CPU: DONE 10-30 03:18 | **X59-S** (human vs self-play data; 15 matched students, 5 seeds each): **SELF-PLAY-DATA-BETTER.** Δ_data = S-H − S-S −7.28 pp [−10.29, −4.28] on U vs Kakuna (S-H 0.320, S-S 0.393); in-house −9.06 [−10.98, −7.14]. **OWN-LABELS-BETTER:** human labels −4.50 [−7.42, −1.58] against P\*'s on the same states. 901 units on one engine at the play pin 89c2c2501 (tag x59s-play-89c2c250, a commit off main). README `measurements/x59s_matched_students_2026-10-25/`. | **H-BC dropped.** The closure of the 1300+ slice waits on Δ_restriction (S-P0 − S-P, CPU, playing since 10-30 03:19); the ≥ 1500 follow-up (≈ 1.5 GPU-h, a window on Training Oct's lease) is registered next. |
| CPU: DONE 10-28 18:21 | **D3 look 2**, the terminal read of set mutation: **UNDECIDED, final.** Kakuna, unfamiliar teams +2.33 pp [−0.69, +5.35] (D3 0.494 vs C0 0.471; 4 of 5 seeds positive); in-house −0.45 pp [−1.16, +0.25], NON-INFERIOR. Every game on ONE engine (E1) at the play pin; all registered checks PASS. Record `x35_breakers_stage_2026-10-17/reads/d3_look2.json`. | Set mutation is not adopted as a goal-(b) lever by this read; it stays a candidate ingredient of X63's union. |
| CPU: DONE 10-30 03:06 | **X48 look 1** (per-token-kind FFN experts grown at P\*, + 20M; N4's one-look screen): **NO-GO, "experts not engaged"** (a weak NO-GO). In-house −0.55 pp [−1.32, +0.21] (90 %; S at 2,000 pairs, C0 re-played); Kakuna on forum72 +2.06 pp [−0.06, +4.19], UNDECIDED (X48 0.526 vs C0 0.506). The experts diverged no more than chance drift (D_k / null 0.72–0.93); mean LR ×1.01. Record `x35_breakers_stage_2026-10-17/reads/x48_look1.json`. | Closes per-kind FFN untying at the plateau within 20M, weakly; not width or `D_MODEL`. N4 or N4-P-direct is the orchestrator's choice. |

## 3. The plan, ranked (2026-10-27)

The orchestrator's order after REVIEW_2 of the all-null strategy (`69c7cd34d`). Every item needs its GO.

1. **X48**, production capacity at the plateau: READ 10-30, **NO-GO** (weak, "experts not engaged"; §2). The next capacity read (N4 with its review's fixes, or N4-P-direct) is the orchestrator's choice.
2. **X63**, a long union continuation (REGISTERED 2026-10-29, not launched;
   [`endstate/design_x63_long_breadth_continuation.md`](endstate/design_x63_long_breadth_continuation.md),
   PREREG [`x63_2026-10-29/`](research_state/measurements/x63_2026-10-29/PREREG.md)): X26 @380M + 100M with 80 % of
   teams from a 17,876-team union (X62's recombined sixes 0.537 of draws, N1-P's procedural teams at its 3,000-team
   tier 0.134, D3's 0.129), read on forum72 against Kakuna at the shared play pin against X26c and X63-C (both trained:
   a 1-arm-v-2-control screen that also measures the +100M seed spread). ≈ 20 GPU-h to its screen; it takes the GPU
   after N1-P look 2's seeds, on the orchestrator's GO.
3. **PR-1′**, the precision read re-scoped by REVIEW_2: D3 and N1-P against C0 on F (D1 reported only), 30,000–38,000
   new Kakuna games at the frozen X35-frame shared pin `x35-shared-play-86cfdf7e` (E1, post-H22; reads.md §4). Information only; it gates nothing above it. X48's read has landed (10-30), so it can run now,
   or at once on a rented CPU box (§6).

**Gated.** **X62**, P\*-forks on 12,000 recombined teams (REVISED DRAFT, `58cb02dc6`): its first gate, H-CTX, read
SUPPORTED, so its corpus `x62_recomb_v1` is built (2026-10-29, as X63's largest component); then PR-1′ KILL retires the fork (N1-P read NULL, so it does not make X62 redundant). Per
resource-day it ranks below X63, and X63 Stage A runs before X62 under both SUPPORTED and UNDECIDED (ruled 2026-10-27). **X53-N1** is not triggered: N1-P read NULL (10-30), not GO. **S-U** (one-ply search on unfamiliar teams, registered `ffd9a443a`) has two BLOCKERs from its review
(`7aafa527f`): where its search rule departs from the policy at P\* (40 % of decisions) it picks the better move
0.513 [0.447, 0.580] of the time, by our own playouts. Its sign-accuracy gate read **GO** (2026-10-28) at 0.635, but
in the states where the true battle seed leaks (the leak's value, not its exposure, is higher on unfamiliar teams). The
leak-reduced re-gate (`dice="rolls"`) supersedes it and **still reads GO**: on unfamiliar teams the departures are right
0.599 [0.543, 0.649] of the time, still by our own playouts
([`su_regate_rolls_2026-10-28/`](research_state/measurements/su_regate_rolls_2026-10-28/README.md)). Next is the
**deviation-only live read**, REGISTERED 2026-10-28 and not yet played
([`dlive_read_2026-10-28/PREREG.md`](research_state/measurements/dlive_read_2026-10-28/PREREG.md)): at ≈ 1,800 such
departures on unfamiliar teams, the departure is forced in a replay of the game against the real Kakuna and compared with
the game as played. GO (lower bound of the mean gain > 0) licenses the amended S-U. ≈ 40 CPU-h; it queues behind X59-S's Δ_restriction.

**Dropped or retired.** **X64**, the toy context-count curve: DROPPED by its review (`c0fd72bd8`; it would cost more
than the production reads it was meant to gate). **N2**, the set-identity lever: dropped, P\*'s encoding is mostly
compositional (`fd4160d76`). **PR-1 as first written**: replaced by PR-1′ (`69c7cd34d`). **rc72**: retired before any
game counted, replaced by rc72b (`905c29f5b`). **The 20M stack fork**: superseded by X63.

## 4. Results of the last week (2026-10-20 → 10-27)

| lever / read | verdict | effect [95 % interval] | record |
|---|---|---|---|
| A1: archive opponent pool | NULL, final | R1 +0.19 pp [−1.07, +1.46] | `806e50931` |
| SPRT: promote snapshots by a sequential test | NULL, final | R1 −0.25 pp [−1.65, +1.16] | `4290d652a` |
| X47: toy depth 2 → 4 | UNDECIDED, Δ̂ ≤ 0, ended | −0.88 pp [−3.20, +1.44] | `d6e57f2e9` |
| X52: practise 36 teams for 5M (ceiling of practice) | UNDECIDED | practised +3.2 pp [−0.2, +6.8]; unpractised +1.3 [−3.5, +6.1] | `42bb0858d` |
| D1: widen the team pool | UNDECIDED, final | Kakuna, unfamiliar teams: +2.10 pp [−0.92, +5.11]; in-pool −1.61 [−2.90, −0.32] | `eb99ab362` |
| X26c: train on 380M → 480M | FLAT; goal (b) not met | +0.08 pp [−3.90, +4.06] | `204204bbf` |
| D3 look 1: set mutation | CONTINUE (look 2 decides). Read ACROSS two engines: D3 on E2, banked C0 on E1. C0's rows also predate the 10-25 observation / label fixes. The guard's 140 / 140 identical games (one C0 cell) bound the divergence below about 2.1 % of games | Kakuna, unfamiliar teams: +1.27 pp [−2.63, +5.16]; in-pool −0.24 [−1.10, +0.62], non-inferior | `58287c525` |
| D3 look 2: set mutation, FINAL | UNDECIDED, terminal. Every game on ONE engine (E1); NULL was unreachable at this size | Kakuna, unfamiliar teams: +2.33 pp [−0.69, +5.35]; in-pool −0.45 [−1.16, +0.25], non-inferior | D3's look-2 record |
| Belief prior mix, λ = 0.25 | NOT DETECTED | +1.50 pp [−2.74, +5.74] | `f299a7243` |
| Belief prior mix, λ = 1 (prior only) | NOT DETECTED; H3 not killed | −1.50 pp [−5.47, +2.47] | `71c76dfbb` |
| Human-only Metamon models (FR-1, FR-3) | P\* BEATS all three | SmallRL 0.666 [0.639, 0.692]; LargeRL 0.631 [0.603, 0.658]; LargeIL 0.790 [0.766, 0.812] | `4d8d9eef8`, `df35d1bb0` |
| PastaMittens (Metamon's ensemble) vs Kakuna as the opponent | NOT DETECTED: Kakuna stays the opponent | Δ unfamiliar −0.25 pp [−4.23, +3.73] | `82f273eff` |
| R0: exploitability of P\* | HEADROOM, saturated | best responder +11.1 pp [+8.3, +13.8]; its learned part transfers: +14.5 [+7.6, +21.2] vs C0, against +13.6 [+7.9, +19.1] vs P\* | `c456b6a09`, `1459416fb` |
| E1-0: a real bot as the search referee | UNDECIDED, no relabel | oracle bot − self −0.024 [−0.054, +0.009] | `015f00e20` |
| Hidden-info maturity, plateau step | GAP-SHRINKS NOT DETECTED on the plateau record; frontier static. The interim-bank replication reads GAP-SHRINKS; the record stands | G_F change −0.003 [−0.020, +0.015]; interim −0.030 [−0.047, −0.012] | `e7048bb1d` |
| L1 replication (piKL, 4-turn leaf) | REPLICATES thinly, pooled over the 25M and P\* anchors; post hoc it is all the 25M anchor's | +0.0097 [+0.0025, +0.0169]; at P\* +0.0001 [−0.0097, +0.0101] | `f756019d7` |
| Piloting-gap forensics (banked Kakuna games, descriptive) | the gap shows after turn 10, from level positions; team vs set NOT separated (the per-mon KO drop is the win gap restated: within won / lost games −0.011 / −0.027; review B1) | level at turn 10: 0.609 home72 vs 0.501 U; seen sets only, per-mon KO edge drop +0.093 [+0.004, +0.176] | `c758fcb82` |
| H-MID: does P\*'s critic see the unfamiliar-team deficit? | **CRITIC-MISCALIBRATED-ON-U**: V's in-game view sees none of it (its 3.5 pp set gap is a turn-1 team prior, mostly P72; review M2); at equal V, P\* wins 8.6 pp less on U; V's ordering (AUC) is equal on both sets. Expected of a self-play critic, so it does not place the memorisation in the critic (review M1) | level at turn 10: Δbias +0.076 [+0.016, +0.139] | `f200258f6` (PREREG `2dd522785`, review `90708bff5`) |
| N2 identity test: does P\* key on our sets' exact identity? (CPU probe) | COMPOSITIONAL: lever N2 dropped | familiar / unfamiliar policy shift under a set re-draw 1.19× [1.07, 1.33] (bar 1.5×) | `98bcb1146` (PREREG), `fd4160d76` |
| Plasticity probe: has the plateau trunk stopped bending to new data? (CPU probe, X26 25M → 480M) | NO-LOSS: plasticity loss unlikely; the strategy tree takes the capacity branch | refit speed on P\*'s own labels 0.986× X26 @25M's (bar < 1.10); near-dormant units 0.78×; a neutral target refits 1.11× slower than @25M, all of it paid by 175M | `8bebdfe31` |
| H-CTX: does the familiarity premium go with the TEAM or the SET? (P\* vs Kakuna on rc72b, new sixes of familiar pool sets, with home72 and F on one engine) | **SUPPORTED: the team.** rc72b recovers −0.34 [−2.05, +0.20] of the home premium; X62's first gate passes | rc72b 0.514, F 0.540, home72 0.615; rc72b − home72 −10.1 pp [−14.2, −6.0] | `6f126c4af` |
| H-CTX A-K: is rc72b weaker as a panel under a NEUTRAL pilot? (Kakuna on both sides, rc72b vs home72 + F teams) | **UNDECIDED, detected weaker** (−28 Elo [−51, −5], 35-Elo bar): uniformly, no coherence gradient; post hoc and across engines, a full transfer would make H-CTX's recovered share 0.21 [−0.88, +0.88] | S 0.459 [0.428, 0.493], 1,080 games | `fb84c260f` (PREREG) |

## 5. How we compare with the state of the art

| opponent | what it is | our result | record |
|---|---|---|---|
| **Kakuna** | Metamon's best public model; 142.8M parameters, offline RL on human and self-play data | home72 **0.598** (ahead); unfamiliar **0.487** (≈ even, goal not met) | `3010c81ba` |
| **PastaMittens** | Metamon's ensemble | U 0.484 [0.456, 0.512], home72 0.550 [0.501, 0.598]: no stronger than Kakuna | `82f273eff` |
| **Metamon human-only models** | trained only on human replays (SmallRL 14M, LargeRL 195M, LargeIL 194M parameters) | P\* beats all three (0.63–0.79). Offline RL beats imitation on the same data by about 16 pp. | `4d8d9eef8`, `df35d1bb0` |
| **Foul Play** (search bot; Gen 9 OU champion) | MCTS over a hand-written evaluator | **By proxy only:** Kakuna draws it, 0.490 [0.422, 0.559], and P\* beats Kakuna on home72. **Never read against P\*.** Our old 75M policy lost 0.388 [0.288, 0.497]. | `8cba100f8`, `e00e78da0` |

So: at the state of the art on familiar teams, with a model about 55 times smaller (70 times by the feature extractor
alone), and level with it on unfamiliar teams, an interval that spans 0.50 (the survey words it "behind"; [`research_state/literature/sota_survey_2026-10-26.md`](research_state/literature/sota_survey_2026-10-26.md),
`fd3a99b65`).

## 6. Decisions waiting on the owner

Every open owner question, in one place.

1. **A cloud GPU, our second card (vast.ai).** The approval asked: top up the credit by $15, spend at most $20 in
   total, one instance at a time, and upload P\* and its pool snapshots (about 170 MB, never human data or
   credentials). Each cloud fork frees 12–13 local GPU-h for about $5. The question is verbatim in
   [`ops/cloud_gpu_plan.md`](ops/cloud_gpu_plan.md) §6 (`37de8515f`); its first workload named D3's look-1 seeds, which
   have since trained locally, so part (c) needs a new workload. X53-N1 would need a wider approval (about $50–85, a
   longer deadman, possibly two instances at once: `endstate/design_x53_n1_fresh_breadth.md` §9). Recommendation: yes.
2. **A rented CPU box.** The Kakuna lane, not the GPU, sets when anything is read: PR-1′ alone is 30,000–38,000
   games, weeks on the three agent cores. A 32-vCPU box would play them in about 8–24 box-hours for about $5–25
   (UNVERIFIED: no rented CPU has played a Kakuna game yet). The same approval would serve X62's, X63's and S-U's
   reads (`endstate/strategy_if_levers_null_2026-10-26.md` §3.1; REVIEW_2 `69c7cd34d`).
3. **The history rewrite.** It replaces human account names with salted codes in every past version of every file,
   then replaces `main`'s history on GitHub. It is built as a dry-run kit (`39c60355c`,
   [`ops/history_rewrite.md`](ops/history_rewrite.md)) and **not run**. The owner's part: the go; four rulings from its
   "Decisions first" section (purge the 61 flagged accounts or all 77, recommended 77; whether the three waived
   identities in a test fixture, the forum corpus and design prose are scrubbed first; whether frozen files such as
   the ledger stay on old SHAs, the default; and when the local old objects are pruned); and a GitHub Support
   request to purge cached copies of the old commits (only the repository owner can file it). Every commit gets a
   new SHA, so this page's gate fails until its SHAs are re-mapped from the table the rewrite writes.
4. **The exchange rate r, for X62 and X63: DECIDED, r = 1** (2026-10-29; the owner: "You choose"; X63's AM-X63-R,
   recorded before X63-U launches). Both breadth designs price their in-pool cost against their unfamiliar-team gain
   against Kakuna: X62's net = Δ_home72 + r · Δ_U; X63's net = Δ_home72 + r · Δ_F. A lever whose net is ≥ 0 is
   adopted. r = 1 because both goals measure the same win rate against the same opponent, and goal 2a has about 5 pp of
   margin over its bar while goal 2b is about 4 pp short. It is never changed after a launch.

## 7. The biggest risks and unknowns

1. **No lever we have measured closes goal (b)'s ≈ 4 pp alone.** D1, D3, X52 and the belief mix each read +1 to
   +3 pp with intervals ±3–4 pp wide. Kakuna games run on CPU at 160–230 an hour on the shared agent cores, so each
   look costs days. Breadth at 10⁴ procedural teams read NULL at the plateau (N1-P, 10-30), so
   [X53-N1](endstate/design_x53_n1_fresh_breadth.md) is not triggered. The bets left are recombined teams (X62; H-CTX
   supports them) and a long union continuation (X63). If the levers miss, the fallback is
   [`endstate/strategy_if_levers_null_2026-10-26.md`](endstate/strategy_if_levers_null_2026-10-26.md), revised after
   its [review](research_state/measurements/strategy_review_2026-10-26/REVIEW.md) (`6fca2c061`) and re-ranked by
   [REVIEW_2](research_state/measurements/strategy_review_2026-10-26/REVIEW_2.md) (`69c7cd34d`, §3 above).
2. **The meter itself.** It is unverified whether Kakuna's self-play saw near copies of our unfamiliar-team panel
   (assumption A-K, open). Foul Play has never been read against P\*. In-house meters are blind to the deficit, so
   only slow Kakuna reads steer goal (b).
3. **The plateau may be the recipe, not the ceiling.** Plain training is FLAT and the population levers are NULL,
   yet a best responder still finds +11 pp. Search does not pay at P\*, and the plateau trunk has not lost its
   plasticity. Holding the learning rate up did not help (LR-F, UNDECIDED final, −1.13 pp [−2.45, +0.18]), and per-kind FFN
   experts gave no ≥ 2 pp in-house gain (X48 NO-GO, weak: the experts barely moved). N4 (FFN width) is the next test
   of whether P\* is the most this architecture can reach.
4. **The X35 family trains under the legacy packer.** `f4b25238` and its corpus-only pins predate `d8d674b99` and
   launch without the pack oracle, so a new corpus can bring Frustration sets that play at base power 1, unseen.
   Forks against a banked `f4b25238` control stay on that family, but only if the PREREG records the corpus's
   material share. Above 2 % of its teams, the arm needs a packer-backport pin and a matched control. Fresh runs
   train at a post-fix pin.
