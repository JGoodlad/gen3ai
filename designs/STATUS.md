# STATUS — where gen3ai stands, on one page

**As of 2026-11-06.** This page is the dashboard: what we are trying to do, where each goal stands, and what is
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
| 1 | **The strongest baseline** | **AT ITS PLATEAU.** P\* = X26 @ 280M. Training on to 480M changed nothing. Holding the learning rate up (LR-F) did not help in-house: UNDECIDED, final, leaning negative. Per-kind FFN experts (X48) gave no ≥ 2 pp in-house gain: NO-GO, weak (the experts barely moved). On the toy, a 4× wider trunk FFN (N4) gave none either: NO-GO. | P\* vs the production baseline: 0.694 [0.666, 0.721]. The 480M continuation vs P\* against Kakuna: +0.08 pp [−3.90, +4.06], FLAT. X48 − C0 on S: −0.55 pp [−1.32, +0.21] (90 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/x48_look1.json`). LR-F − C0 on S: −1.13 pp [−2.45, +0.18] (97.5 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/lrfloor_look1.json`). | `d172c10ed`, `204204bbf`, `f7fddc39b`, `1355d2d3f`, `b01dc048b` |
| 2a | **Beat Metamon on our teams** | **MET.** | P\* vs Kakuna on home72: **0.598 [0.549, 0.644]** | `3010c81ba` |
| 2b | **Beat Metamon on unfamiliar teams** | **NOT MET**, about 4 pp short. The lower bound must clear 0.50, which needs about 0.53. | P\* 0.487 [0.459, 0.515] (its forum72 third under Kakuna's legacy pack, below). No deployable lever's interval clears 0.50. The leak-inclusive search S′ does (S′-LIVE: 0.545 [0.526, 0.564] on 2,698 fresh U deals, raw P\* 0.498 on the same deals), quoted only as "GOAL-B-REACHED (leaky)": S′ sees the true battle seed. Highest point estimates, each against its own control and none detected: the belief prior mix (λ = 0.25) 0.502, D1 0.492, D3 0.494 (final, UNDECIDED). N1-P (look 2, final: NULL) 0.500 is M4-hazarded (its corpus overlaps H and P72) and is not read as goal (b); on F it reads +0.55 pp [−1.26, +2.36] over C0. Human data is no lever so far: a student on 1300+ human rows plays U 7.28 pp [4.28, 10.29] worse against Kakuna than one on P\*'s self-play rows (X59-S, `c1d2d7fcb`); that closes nothing (`5bbba6096`), and the ≥ 1500 follow-up reads SELF-PLAY-DATA-BETTER too (−4.42 pp [−6.07, −2.76] in-house), with rating NOT-RESOLVED (`71ef78a79`, §2). | `3010c81ba`, `f299a7243`, `eb99ab362`, `25a736170`, `62e46874a`, `c1d2d7fcb`, `5bbba6096` |
| 3 | **Know where we are** | **MAPPED.** The deficit is **piloting** an unfamiliar team, not facing one. In-house meters cannot see it; only Kakuna games can. Nothing measurable is lost in the opening; the gap shows from level positions after turn 10 (descriptive). It does not go with the SETS (H-CTX SUPPORTED): on new sixes of familiar pool sets P\* scores 0.514, below F's 0.540 and 10 pp under home72 on the same engine; the sets keep at most about a fifth of the premium. Why the six matters (memorised sixes, broken cores, no plan) is open (`hctx_rc72_2026-10-26/REVIEW.md` M1). Under a neutral pilot (Kakuna on both sides) rc72b is only about 4 pp weaker, flat across coherence (`hctx_kk_2026-10-27`), so the premium stays mostly team-level. P\*'s own critic does not see it (H-MID: CRITIC-MISCALIBRATED-ON-U), as a self-play critic cannot (REVIEW M1). | Piloting deficit, strength-corrected (X26 @161M, PROVISIONAL, resting on the unverified assumption A-K): +0.123 [+0.039, +0.206], facing +0.017 [−0.062, +0.095]; their difference +0.106 [−0.009, +0.221] is UNDECIDED. P\* is level with Metamon's ensemble on unfamiliar teams, ahead of it on home72, and beats every human-only Metamon model (§5). | `8b43eb44a`, `82858f433`, `c758fcb82`, `905c29f5b`, `f200258f6`, `90708bff5`, `82f273eff`, `df35d1bb0` |
| 4 | **AlphaStar-style improvement** (league, PFSP, exploiters) | **NO GAIN YET.** The archive-pool and promotion levers are NULL. An exploiter beats P\* easily, so headroom exists. No league arm has run. | Best responder vs P\*: **+11.1 pp [+8.3, +13.8]** (saturates by 8M steps) | `806e50931`, `4290d652a`, `c456b6a09` |
| 5 | **What search needs** | **A LEAK-INCLUSIVE GO against the real bot (S′-LIVE, 2026-11-05).** P\* + the search S′, deciding every move from turn 10, beats P\* alone against Kakuna on unfamiliar teams. S′ sees the true battle seed (speed ties, hidden counters), so this is not deployable search; it licenses only a leak-honest replication, and that work alone is un-paused. Graded by our own model alone, earlier search gains did not survive (L1 at P\*). Distilling S′ into P\* by one supervised pass on its banked choices did not deliver them (X65, NOT-DELIVERED, 11-06). | S′-LIVE G_U **+4.60 pp [+2.66, +6.48]** (2,698 paired deals; worst-case-imputed LB +2.57); searched-arm U score 0.545 [0.526, 0.564], "GOAL-B-REACHED (leaky)" at most. L1 cell at P\*: +0.0001 [−0.0097, +0.0101]. | `9c2f023e4`, `d93a51124`, `f756019d7`, `015f00e20`, `ff6874fcc` |
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

## 2. Running now (as of 2026-11-06)

Read at 03:20 PST on 2026-11-04 from `python -m main.ops.arm_chain status` (each chain file under
`~/.claude/jobs/x35/chains/`), `nvidia-smi` and `scripts/ops/mem_cap.sh --status`. Only the GPU and the N4 chain
were re-read on 11-04, and the X65 row on 11-06 01:10; nothing else here was re-checked.

| where | job | what decision it drives |
|---|---|---|
| GPU: IDLE (11-04 03:19) | The N4 chain `n4-toy-2026-11-03` is DONE: all four re-run arms trained and recorded (K9(b) PASS + dose); its supervisor exited 0 at 01:13. | N4 read NO-GO (below). |
| GPU: not read | **X63-U s6302** (Stage B's speculative seed) is `DONE_FAILED` in its chain (`x63u-stageb-spec-oct30`, 10-31 00:02). Stage A read NOT PURSUED, so it would not have been read for any decision anyway. | None. |
| CPU (11-06 01:10) | **X65 ExIt-0 is DONE, NOT-DELIVERED**: the delivery gate ran on the CPU (≈ 7 min); no Kakuna game was played, and the look-1 state is planned and unplayed. | ExIt-1 not funded; X60-T is the GPU's next candidate. |
| CPU: IDLE (11-05 16:47) | **S′-LIVE is DONE**: its last unit ended 15:37, the identity re-score PASSed, the record was written 16:47. | Search: GO-LEAKY (below). |

### Landed since 2026-10-27

| when | job | result | what it decided |
|---|---|---|---|
| CPU 11-06 01:05 PST | **X65 ExIt-0: NOT-DELIVERED** (PREREG Amendments `df7288a04`, `42dff4e9d`, `ff6874fcc`). Three P\* students, each fine-tuned toward S′'s piKL targets on two panels' banked decisions, agree with S′'s pick on the panel they never saw LESS than P\* does: pooled Δagree **−1.26 pp [−1.40, −1.11]** (H −1.06, F −0.94, p72 −1.78; 97,395 label rows). By hand equal; P\*'s offline forward reproduces its recorded choice on 100 % of rows. No Kakuna game was played ([`x65_exit0_2026-11-05/`](research_state/measurements/x65_exit0_2026-11-05/README.md)). | One supervised distillation pass at this recipe does not deliver S′'s choices; it does not close ExIt (honest labels, larger budgets, in-loop forms). | ExIt-1 is not funded; X60-T is the GPU's next candidate. |
| CPU 11-05 16:47 PST | **S′-LIVE: GO-LEAKY** (PREREG `9c2f023e4`, AM-K6 `d93a51124`). P\* + S′ (`r0s4_rolls@10`, every eligible decision from turn 10) against P\* alone, both against the real Kakuna, on 2,698 fresh paired unfamiliar-team deals: **+4.60 pp [+2.66, +6.48]** (worst-case-imputed LB +2.57; H +4.5, F +5.2, p72 +4.1). The searched arm scores **0.545 [0.526, 0.564]** (raw 0.498). Every guard PASS (G-PREFIX 0 mismatches, identity re-score 600 / 600, 2 post-treatment drops). ≈ 315 CPU-h, 102 h over four supervisor lives ([`sprime_live_play_2026-11-01/`](research_state/measurements/sprime_live_play_2026-11-01/README.md)). | Goal (b) is "REACHED (leaky)" at most: S′ sees the true battle seed. | Licenses ONLY a leak-honest replication on fresh deals; no training, league or public-server change. |
| CPU 11-04 03:14 PST | **N4 look 1**, the toy capacity screen (T\* with every trunk FFN grown 256 → 1,024, +70M, against a same-pin C0′; 3 + 3 seeds; PREREG `8ed8164ea`, AM-N4-FIXES `801a96d0e`, NOTE 2 `6aa46f9e0`): **NO-GO.** In-house **+0.46 pp [−0.28, +1.20]** (90 %; S at 2,000 pairs per cell); Kakuna on forum72 **+0.20 pp [−2.91, +3.31]**, UNDECIDED (N4 0.431 vs C0′ 0.429, 3,000 games each). The new units were used, at about the scale of ordinary 70M drift (registered reading "used, buys nothing"). Every registered check PASS; every row at `31bd9ced9`. Survived a GPU fault (11-01) and a box reboot (11-03): the four remaining arms re-ran under the same names and pin. README `research_state/measurements/n4_capacity_screen_2026-10-27/`. | Closes growth at the plateau on the toy within 70M (depth X47, FFN width N4). | No N4-P on toy evidence; the scale bet loses its cheap support (not refuted). |
| CPU 11-03 11:59 PST | **X59-S's ≥ 1500 follow-up** (in-house only; 15 fresh students at 270,260 rows each; 30 cells of 200 pairs on the read's one engine, played from the pin tag x59s-play-08271d24 (a commit off main) under Amendment 8 because the registered pin could not load the v164 students; `71ef78a79`). **Rating: NOT-RESOLVED**, S-H≥1500 − S-H-mix **+1.33 pp [−2.76, +5.43]**. **Data: SELF-PLAY-DATA-BETTER**, S-H≥1500 − S-S cut **−4.42 pp [−6.07, −2.76]**. Read as rating at a fixed slice and volume: the same panel-excluded, forfeit-heavy slice, with concentrated accounts and shared rows. | Higher-rated human rows still teach a fresh network less than our self-play rows, and the rating effect is unresolved. | Routes nothing alone; human data's closure stays blocked (Δ_restriction), H-TEAMS (X60-T) stays first. |
| CPU 11-01 08:09 PST | **The deviation-only live read: STOP.** At 1,793 departures of the search S′ (`r0s4_rolls@10`) from P\* on unfamiliar teams, S′'s pick was forced in a replay of the game against the real Kakuna. It won **+1.20 pp [−1.55, +3.88]** a departure more than the game as played (H +4.2, F +0.5, p72 −1.1; home72 +2.7 [−4.0, +9.6], reported). The self-referee's own edge on the same fresh departures is +0.99 pp, against +3.4 on the gate's states. Every registered check passes: one engine, 135 / 135 null replays identical, 9 / 2,100 branches dropped. All 9 drops are Kakuna replay divergences, made drops instead of a VOID by the blind AM-DL-PEER (`e40d9663c`, `276c5b7e6`). 36.6 CPU-h. README `measurements/dlive_read_2026-10-28/README.md`. | Search stays paused for goal (b). | No amended S-U on S′. |
| CPU 10-31 16:12 | **X63 Stage A: NOT PURSUED.** X63-U s2601 (X26@380M + 100M, 80 % of teams from the 17,876-team union) against X26c and X63-C on forum72 vs Kakuna: Δ_F **−0.69 pp [−3.75, +2.37]** (X63-U 0.512, X26c 0.521, X63-C 0.517; 4,000 games each), short of the +1.0 pp screen; home72 −1.05 [−4.62, +2.52] (reported). σ̂_CC = 0: the two controls differ by 0.43 pp, inside binomial noise. 18,000 games on ONE engine (E1) at the play pin 65b998da (tag x63-play-65b998da); every registered check PASS. Record `measurements/x63_2026-10-29/reads/stageA.json`. | The union continuation at share 0.8 gives no ≥ +1 pp F gain at one seed; a true +3 pp is outside the interval. It claims nothing for the era. | **Stage B does not fire**; s6302 is not read. P(NOT PURSUED) at a true +2 / +3 pp was 0.149 / 0.020 (σ = 0 row). |
| GPU 10-30 14:45 | **X63-U s2601 TRAINED, clean** (1,269.9 min, 1,339 steps/s): K9(b) PASS (AM-6) and its dose recorded (`695b20b8c`). X63-C s6302, the control replicate, had trained earlier: K9(b) PASS, 1,242.9 min (`8a7f293f3`). X26c is the banked control. | Both arms of Stage A exist; no X63 score yet. | X63 is launched with r = 1 (AM-X63-R, `13084395e`, §6) and its canary rule AM-X63-CANARY (`71afc431e`: a FATAL made only of slice-0/2 per-parameter bf16 refusals, resumed at the pin, does not count toward the restart limit; at most two per seed). |
| GPU 10-30 14:46-15:19 | **Canary slice-2 diagnosis** (`6b3966dc6`; verdict `7775b2241`): the compiled-numerics hypothesis is **RULED OUT at 5 states**; every state reads CONDITIONING (compiled median distance 0.09-0.55 of the fp32 dither null's maximum). The 30 GPU slow-tier rows all PASS (`1a07fd819`). | The refusals are an ill-conditioned single-row-carried parameter against a lucky eager denominator, not a compiler fault. | The bf16 per-parameter bar now divides by the fp32 dither null (`gen3_canary_param_null_v1`) **at new pins only**; runs pinned earlier keep the eager bar. The earlier stage-2 read stays **INCONCLUSIVE**: the row bar's cap stays 0.72 (`designs/training/canary_recalibration_2026-10-28.md`). |
| CPU 10-30 17:24 | **X59-S, final**: SELF-PLAY-DATA-BETTER. Δ_data = S-H − S-S −7.28 pp [−10.29, −4.28] on U vs Kakuna (S-H 0.320, S-S 0.393); in-house −9.06 [−10.98, −7.14]. OWN-LABELS-BETTER: human labels −4.50 [−7.42, −1.58] against P\*'s on the same states (`c1d2d7fcb`). 901 units on one engine at the play pin 89c2c2501 (tag x59s-play-89c2c250, a commit off main). README `measurements/x59s_matched_students_2026-10-25/`. | The closure is **BLOCKED**: Δ_restriction (S-P0 − S-P) +2.85 pp [+0.14, +5.56] (`5bbba6096`); S-P0 = S-S (0.393), descriptive. | **H-BC dropped.** SELF-PLAY-DATA-BETTER closes nothing, so human data stays open and routes H-TEAMS first (X60-T). The ≥ 1500 follow-up read 11-03 (the row above). |
| CPU 10-30 11:04 | **N1-P look 2** (X61P, final), the five forks with half their teams from 12,000 procedural teams: **NULL.** F +0.55 pp [−1.26, +2.36] (N1-P 0.509 vs C0 0.504, 6,000 forum72 games per arm; seeds +1.0 / −0.25 / +1.5 / +1.0 / −0.5), inside ±3 pp; in-house S on home72 −3.28 pp [−4.14, −2.43] (INFERIORITY-NOT-EXCLUDED). Every game on ONE engine (E3) at look 1's commit (AM-N1P-2, `d78700253`). Record `62e46874a`. | Rules out **"N1 at the plateau"** (share 0.5, 20M) only: not N1 from step 0, the toy screen or a human-derived corpus. | X53-N1's trigger (N1-P GAIN) did not fire. X63 unchanged (AM-X63-N1L2: NULL ⇒ unchanged). |
| CPU 10-30 03:06 | **X48 look 1** (per-token-kind FFN experts grown at P\*, + 20M; N4's one-look screen): **NO-GO, "experts not engaged"** (a weak NO-GO). In-house −0.55 pp [−1.32, +0.21] (90 %; S at 2,000 pairs, C0 re-played); Kakuna on forum72 +2.06 pp [−0.06, +4.19], UNDECIDED (X48 0.526 vs C0 0.506). The experts diverged no more than chance drift (D_k / null 0.72-0.93); mean LR ×1.01. Record `b01dc048b`; its slope group was not resumed (`242f74d62`, reported only). | Closes per-kind FFN untying at the plateau within 20M, weakly; not width or `D_MODEL`. | N4 or N4-P-direct is the orchestrator's choice. |
| CPU 10-28 18:21 | **D3 look 2**, the terminal read of set mutation: **UNDECIDED, final.** Kakuna, unfamiliar teams +2.33 pp [−0.69, +5.35] (D3 0.494 vs C0 0.471; 4 of 5 seeds positive); in-house −0.45 pp [−1.16, +0.25], NON-INFERIOR. Every game on ONE engine (E1) at the play pin; all registered checks PASS. Record `25a736170`. | Set mutation is not adopted as a goal-(b) lever by this read. | It stays a candidate ingredient of X63's union. |

## 3. The plan, ranked (as revised 2026-10-30)

The orchestrator's order after REVIEW_2 of the all-null strategy (`69c7cd34d`), with the verdicts since 10-27 applied. Every item needs its GO.

**X65 ExIt-0, search distillation: READ 2026-11-06, NOT-DELIVERED** ([`endstate/design_x65_exit0_search_distillation.md`](endstate/design_x65_exit0_search_distillation.md); §2).
- The students did not learn S′'s choices on unseen teams (pooled Δagree −1.26 pp [−1.40, −1.11]), so no game
  was played and ExIt-1 (≈ 35 GPU-h) is not funded. X60-T is the GPU's next candidate.
- Open, each needing its own registration: ExIt-0H on S′-HONEST's honest labels; a stronger distillation recipe.

1. **Capacity at the plateau: CLOSED for growth.** X48 (production, per-kind FFN experts) READ 10-30, **NO-GO** (weak, `b01dc048b`); the N4 toy (FFN width 4×) READ 11-04, **NO-GO** ("used, buys nothing", §2). No N4-P on toy evidence; `D_MODEL` and from-scratch width (the scale bet) stay open, unqueued.
2. **X63**, a long union continuation: **Stage A read NOT PURSUED (10-31)**, so X63 stops at Stage A
   ([`endstate/design_x63_long_breadth_continuation.md`](endstate/design_x63_long_breadth_continuation.md), PREREG
   [`x63_2026-10-29/`](research_state/measurements/x63_2026-10-29/PREREG.md), read
   [`reads/`](research_state/measurements/x63_2026-10-29/reads/README.md)). X26 @380M + 100M with 80 % of teams from a
   17,876-team union scored Δ_F −0.69 pp [−3.75, +2.37] against X26c and X63-C on forum72 vs Kakuna, short of the
   +1.0 pp screen; σ̂_CC = 0. Stage B does not fire, and s6302 is a speculative replicate not read for any decision.
   Its REPORTED groups (the 400M / 430M nodes, H / P72 / rc72b, P\*) were never played (AM-X63-SPEC FINDING 11).
3. **PR-1′**, the precision read re-scoped by REVIEW_2: D3 and N1-P against C0 on F (D1 reported only), 30,000–38,000
   new Kakuna games at the frozen X35-frame shared pin `x35-shared-play-86cfdf7e` (E1, post-H22; reads.md §4). Information only; it gates nothing above it. X48's read has landed (10-30), but the agent cores are on the dlive raw lane and X59-S's follow-up cells (§2), so it can run only after them,
   or at once on a rented CPU box (§6).

**Gated.** **X62**, P\*-forks on 12,000 recombined teams (REVISED DRAFT, `58cb02dc6`): its first gate, H-CTX, read
SUPPORTED, so its corpus `x62_recomb_v1` is built (2026-10-29, as X63's largest component); then PR-1′ KILL retires the fork (N1-P read NULL, so it does not make X62 redundant). Per
resource-day it ranked below X63, which ran first (ruled 2026-10-27) and read NOT PURSUED at Stage A (10-31); its rank is now the orchestrator's. **X53-N1** is not triggered: N1-P read NULL (10-30, `62e46874a`), not GO. **S-U** (one-ply search on unfamiliar teams, registered `ffd9a443a`) has two BLOCKERs from its review
(`7aafa527f`): where its search rule departs from the policy at P\* (40 % of decisions) it picks the better move
0.513 [0.447, 0.580] of the time, by our own playouts. Its sign-accuracy gate read **GO** (2026-10-28) at 0.635, but
in the states where the true battle seed leaks (the leak's value, not its exposure, is higher on unfamiliar teams). The
leak-reduced re-gate (`dice="rolls"`) supersedes it and **still reads GO**: on unfamiliar teams the departures are right
0.599 [0.543, 0.649] of the time, still by our own playouts
([`su_regate_rolls_2026-10-28/`](research_state/measurements/su_regate_rolls_2026-10-28/README.md)). The **deviation-only live read** (registered `715e0b482`; [`dlive_read_2026-10-28/`](research_state/measurements/dlive_read_2026-10-28/README.md)) then asked the real Kakuna and read **STOP** (2026-11-01).
- **The result:** at ≈ 1,800 such departures on unfamiliar teams, forcing the departure in a replay of the game wins +1.20 pp [−1.55, +3.88] more than the game as played.
- **The self-referee on those fresh departures:** only +0.99 pp, against +3.4 on the gate's states. So the gate's headroom mostly did not carry over, and the read had little power at that size.
- **What follows:** S-U is not played on S′, and search stays paused.
- **Then S′-LIVE read GO-LEAKY** (2026-11-05; [`sprime_live_play_2026-11-01/`](research_state/measurements/sprime_live_play_2026-11-01/README.md)). It asked the game-level question directly: S′ playing every eligible decision against the real Kakuna on 2,698 fresh unfamiliar-team deals, paired with P\* alone, wins **+4.60 pp [+2.66, +6.48]**, and the searched arm scores 0.545 [0.526, 0.564]. It is leak-inclusive by name (S′'s worlds replay the true seed), so it licenses only a leak-honest replication, never "goal (b) met". dlive's per-departure STOP stands as written; the two are different estimands, never pooled.

**Dropped or retired.** **X64**, the toy context-count curve: DROPPED by its review (`c0fd72bd8`; it would cost more
than the production reads it was meant to gate). **N2**, the set-identity lever: dropped, P\*'s encoding is mostly
compositional (`fd4160d76`). **PR-1 as first written**: replaced by PR-1′ (`69c7cd34d`). **rc72**: retired before any
game counted, replaced by rc72b (`905c29f5b`). **The 20M stack fork**: superseded by X63.

## 4. Results of the last two weeks (2026-10-20 → 10-31)

| lever / read | verdict | effect [95 % interval] | record |
|---|---|---|---|
| A1: archive opponent pool | NULL, final | R1 +0.19 pp [−1.07, +1.46] | `806e50931` |
| SPRT: promote snapshots by a sequential test | NULL, final | R1 −0.25 pp [−1.65, +1.16] | `4290d652a` |
| X47: toy depth 2 → 4 | UNDECIDED, Δ̂ ≤ 0, ended | −0.88 pp [−3.20, +1.44] | `d6e57f2e9` |
| X52: practise 36 teams for 5M (ceiling of practice) | UNDECIDED | practised +3.2 pp [−0.2, +6.8]; unpractised +1.3 [−3.5, +6.1] | `42bb0858d` |
| D1: widen the team pool | UNDECIDED, final | Kakuna, unfamiliar teams: +2.10 pp [−0.92, +5.11]; in-pool −1.61 [−2.90, −0.32] | `eb99ab362` |
| X26c: train on 380M → 480M | FLAT; goal (b) not met | +0.08 pp [−3.90, +4.06] | `204204bbf` |
| D3 look 1: set mutation | CONTINUE (look 2 decides). Read ACROSS two engines: D3 on E2, banked C0 on E1. C0's rows also predate the 10-25 observation / label fixes. The guard's 140 / 140 identical games (one C0 cell) bound the divergence below about 2.1 % of games | Kakuna, unfamiliar teams: +1.27 pp [−2.63, +5.16]; in-pool −0.24 [−1.10, +0.62], non-inferior | `58287c525` |
| D3 look 2: set mutation, FINAL | UNDECIDED, terminal. Every game on ONE engine (E1); NULL was unreachable at this size | Kakuna, unfamiliar teams: +2.33 pp [−0.69, +5.35]; in-pool −0.45 [−1.16, +0.25], non-inferior | `25a736170` |
| LR-F look 1: hold the learning rate up | UNDECIDED, FINAL, leaning negative | in-house S: −1.13 pp [−2.45, +0.18] (97.5 %, 3 seeds) | `1355d2d3f` |
| X48 look 1: per-kind FFN experts at P\* | NO-GO, weak ("experts not engaged") | in-house dS −0.55 pp [−1.32, +0.21]; Kakuna on forum72 +2.06 [−0.06, +4.19], UNDECIDED | `b01dc048b` |
| N1-P look 1: procedural breadth at the plateau | CONTINUE | F +0.75 pp [−1.56, +3.06]; P −3.27 [−4.20, −2.33] | `06dba8230` |
| N1-P look 2, FINAL (one engine, E3) | NULL | F +0.55 pp [−1.26, +2.36]; in-house S −3.28 [−4.14, −2.43] | `62e46874a` |
| X63 Stage A: +100M on a 17,876-team union at share 0.8 (one engine, E1) | NOT PURSUED (screen +1.0 pp) | F −0.69 pp [−3.75, +2.37]; home72 −1.05 [−4.62, +2.52]; σ̂_CC 0 | `x63_2026-10-29/reads/stageA.json` |
| X59-S: human vs self-play data (15 matched students) | SELF-PLAY-DATA-BETTER; OWN-LABELS-BETTER; H-BC dropped; the closure BLOCKED | U vs Kakuna Δ_data −7.28 pp [−10.29, −4.28]; own labels −4.50 [−7.42, −1.58]; Δ_restriction +2.85 [+0.14, +5.56] | `c1d2d7fcb`, `5bbba6096` |
| X59-S's ≥ 1500 follow-up (15 students, in-house) | rating NOT-RESOLVED; ≥ 1500 data SELF-PLAY-DATA-BETTER | S-H≥1500 − S-H-mix +1.33 pp [−2.76, +5.43]; S-H≥1500 − S-S cut −4.42 [−6.07, −2.76] | `71ef78a79` |
| S-U gate: sign accuracy of S's departures from P\* (self-referee) | GO (pooled U, with the leak caveat); the leak-reduced re-gate (`dice="rolls"`) also GO | pooled-U 0.635 [0.578, 0.687] (n 346); re-gate 0.599 [0.543, 0.649] (n 362) | `fea88ceeb`, `a61375fa7` |
| The deviation-only live read: S′'s departures forced against the real Kakuna | STOP (search stays paused) | pooled U +1.20 pp a departure [−1.55, +3.88] (n 1,793); self-referee on the same departures +0.99 | `715e0b482`, `276c5b7e6` |
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
| H-CTX A-K: is rc72b weaker as a panel under a NEUTRAL pilot? (Kakuna on both sides, rc72b vs home72 + F teams) | **UNDECIDED, detected weaker** (−28 Elo [−51, −5], 35-Elo bar): uniformly, no coherence gradient; post hoc and across engines, a full transfer would make H-CTX's recovered share 0.21 [−0.88, +0.88] | S 0.459 [0.428, 0.493], 1,080 games | `79ffe8a49` (PREREG `fb84c260f`) |

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
   reads (S′-LIVE itself played locally in 102 h, ≈ 315 CPU-h) (`endstate/strategy_if_levers_null_2026-10-26.md` §3.1; REVIEW_2 `69c7cd34d`).
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
   [X53-N1](endstate/design_x53_n1_fresh_breadth.md) is not triggered. The bet left is recombined teams (X62; H-CTX
   supports them); the long union continuation (X63) read NOT PURSUED at Stage A (10-31). If the levers miss, the fallback is
   [`endstate/strategy_if_levers_null_2026-10-26.md`](endstate/strategy_if_levers_null_2026-10-26.md), revised after
   its [review](research_state/measurements/strategy_review_2026-10-26/REVIEW.md) (`6fca2c061`) and re-ranked by
   [REVIEW_2](research_state/measurements/strategy_review_2026-10-26/REVIEW_2.md) (`69c7cd34d`, §3 above).
2. **The meter itself.** It is unverified whether Kakuna's self-play saw near copies of our unfamiliar-team panel
   (assumption A-K, open). Foul Play has never been read against P\*. In-house meters are blind to the deficit, so
   only slow Kakuna reads steer goal (b).
3. **The plateau may be the recipe, not the ceiling.** Plain training is FLAT and the population levers are NULL,
   yet a best responder still finds +11 pp. Search does not pay at P\*, and the plateau trunk has not lost its
   plasticity. Holding the learning rate up did not help (LR-F, UNDECIDED final, −1.13 pp [−2.45, +0.18]), and per-kind FFN
   experts gave no ≥ 2 pp in-house gain (X48 NO-GO, weak: the experts barely moved). On the toy a 4× wider trunk FFN
   gave none either (N4 NO-GO, +0.46 pp [−0.28, +1.20]): growing the network at the plateau does not move it.
4. **The X35 family trains under the legacy packer.** `f4b25238` and its corpus-only pins predate `d8d674b99` and
   launch without the pack oracle, so a new corpus can bring Frustration sets that play at base power 1, unseen.
   Forks against a banked `f4b25238` control stay on that family, but only if the PREREG records the corpus's
   material share. Above 2 % of its teams, the arm needs a packer-backport pin and a matched control. Fresh runs
   train at a post-fix pin.
5. **The compile canary can refuse a healthy run.** Its bf16 per-parameter rule refused on the confirmation slice 2
   in six of seven recorded refusals (across three rules) and never on slice 1; the 10-30 GPU replay read CONDITIONING at all 5 states, so a
   real compiled-numerics fault is ruled out there (`7775b2241`). The fix (a bar that divides by the fp32 dither
   null) applies to new pins only, so X63-U, pinned on 10-29 at 39eaa326, keeps the eager bar; AM-X63-CANARY
   (`71afc431e`) lets up to two such resumes pass per seed. Stage 2 of the row-bar recalibration is INCONCLUSIVE,
   so the cap stays 0.72.
