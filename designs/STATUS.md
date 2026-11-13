# STATUS — where gen3ai stands, on one page

**As of 2026-11-13.** This page is the dashboard: what we are trying to do, where each goal stands, and what is
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

**The owner's direction (2026-11-09): model-first.** Build the most performant model possible; wherever possible,
improve the model, and making the model a better driver of search counts as improving it. Search that changes no
weight is valued but CPU-side and lower priority (§3).

| # | goal | status | best number [95 % interval] | proof |
|---|---|---|---|---|
| 1 | **The strongest baseline** | **AT ITS PLATEAU.** P\* = X26 @ 280M. Training on to 480M changed nothing. Holding the learning rate up (LR-F) did not help in-house: UNDECIDED, final, leaning negative. Per-kind FFN experts (X48) gave no ≥ 2 pp in-house gain: NO-GO, weak (the experts barely moved). On the toy, a 4× wider trunk FFN (N4) gave none either: NO-GO. | P\* vs the production baseline: 0.694 [0.666, 0.721]. The 480M continuation vs P\* against Kakuna: +0.08 pp [−3.90, +4.06], FLAT. X48 − C0 on S: −0.55 pp [−1.32, +0.21] (90 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/x48_look1.json`). LR-F − C0 on S: −1.13 pp [−2.45, +0.18] (97.5 %, 3 seeds; `x35_breakers_stage_2026-10-17/reads/lrfloor_look1.json`). | `d172c10ed`, `204204bbf`, `f7fddc39b`, `1355d2d3f`, `b01dc048b` |
| 2a | **Beat Metamon on our teams** | **MET.** | P\* vs Kakuna on home72: **0.598 [0.549, 0.644]** | `3010c81ba` |
| 2b | **Beat Metamon on unfamiliar teams** | **MET WITH SEARCH only, AS REGISTERED** (S′-HONEST, 11-09), **and NOT ESTABLISHED once the games are clustered on teams** (the project's own unit of analysis; clustering addendum, 11-09, reported). P\* alone is NOT MET, about 4 pp short (the lower bound must clear 0.50, which needs about 0.53). With the leak-honest search S′h (`r0s4_honest_inc@10`) on every eligible decision from turn 10, P\* clears it under the registered interval, which treats the 5,100 deals as independent: "goal (b) met with search", only ever quoted with that regime AND its team-clustered interval. The search GAIN itself holds under every clustering, but is detected on the procedural panel P72 alone (row 5). No TRAINING lever's interval clears 0.50: highest points, each against its own control and none detected, the belief prior mix (λ = 0.25) 0.502, D1 0.492, D3 0.494; N1-P 0.500 is M4-hazarded and not read as goal (b). Human data is no lever so far (X59-S: SELF-PLAY-DATA-BETTER, −7.28 pp [−10.29, −4.28]; its ≥ 1500 follow-up the same, rating NOT-RESOLVED). | P\* + S′h against Kakuna (checkpoint 34, greedy) on U = H, F, P72: **0.5145 [0.5008, 0.5282]**, 5,100 fresh deals, ties as non-wins (GOAL-B-REACHED as registered). Clustered on our team [0.494, 0.535], on Kakuna's [0.495, 0.534], on both [0.491, 0.538] ([addendum](research_state/measurements/sprime_honest_play_2026-11-06/addendum_clustering/README.md)). P\* alone 0.487 [0.459, 0.515] (X51; 0.4948 on S′-HONEST's deals). | `3010c81ba`, `e81f80657`, `f299a7243`, `eb99ab362`, `25a736170`, `c1d2d7fcb`, `71ef78a79` |
| 3 | **Know where we are** | **MAPPED.** The deficit is **piloting** an unfamiliar team, not facing one. In-house meters cannot see it; only Kakuna games can. Nothing measurable is lost in the opening; the gap shows from level positions after turn 10 (descriptive). It does not go with the SETS (H-CTX SUPPORTED): on new sixes of familiar pool sets P\* scores 0.514, below F's 0.540 and 10 pp under home72 on the same engine; the sets keep at most about a fifth of the premium. Why the six matters (memorised sixes, broken cores, no plan) is open (`hctx_rc72_2026-10-26/REVIEW.md` M1). Under a neutral pilot (Kakuna on both sides) rc72b is only about 4 pp weaker, flat across coherence (`hctx_kk_2026-10-27`), so the premium stays mostly team-level. P\*'s own critic does not see it (H-MID: CRITIC-MISCALIBRATED-ON-U), as a self-play critic cannot (REVIEW M1). | Piloting deficit, strength-corrected (X26 @161M, PROVISIONAL, resting on the unverified assumption A-K): +0.123 [+0.039, +0.206], facing +0.017 [−0.062, +0.095]; their difference +0.106 [−0.009, +0.221] is UNDECIDED. P\* is level with Metamon's ensemble on unfamiliar teams, ahead of it on home72, and beats every human-only Metamon model (§5). | `8b43eb44a`, `82858f433`, `c758fcb82`, `905c29f5b`, `f200258f6`, `90708bff5`, `82f273eff`, `df35d1bb0` |
| 4 | **AlphaStar-style improvement** (league, PFSP, exploiters) | **NO GAIN YET.** The archive-pool and promotion levers are NULL. An exploiter beats P\* easily, so headroom exists. No league arm has run. | Best responder vs P\*: **+11.1 pp [+8.3, +13.8]** (saturates by 8M steps) | `806e50931`, `4290d652a`, `c456b6a09` |
| 5 | **What search needs** | **A LEAK-HONEST GO in games against the real bot (S′-HONEST, 11-09).** P\* + S′h, deciding every eligible decision from turn 10 with worlds built from what our side has seen alone, beats P\* alone against Kakuna on unfamiliar teams. The search program for goal (b) is UN-PAUSED, CPU-side and below the model-first GPU queue (owner, 11-09; §3). About half of the leak-inclusive S′-LIVE point (+4.60, on other deals) did not survive the honest builder (reported, unpaired). Graded by our own model alone, search gains on the banks still do not show at P\* (L1). Distilling S′'s leak-inclusive choices into P\* does not deliver them (X65 NOT-DELIVERED), and the learnability probe reads them unlearnable at this scale (DESCRIPTIVE). | S′-HONEST G_U **+2.00 pp [+0.34, +3.51]** (5,097 paired deals; worst-case-imputed LB +0.32); panels reported: P72 +4.69, H +1.32, F 0.00. Team-clustered, the LB stays above 0 (+0.31 at worst), and the gain is detected on P72 alone: H and F together +0.66 [−1.39, +2.67] (post hoc; P72 is the panel built from the usage tables the search's sampler draws from). S′-LIVE (leaky) +4.60 [+2.66, +6.48]. L1 cell at P\*: +0.0001 [−0.0097, +0.0101]. | `e81f80657`, `f7da935ab`, `9c2f023e4`, `f756019d7`, `0b9e6f354` |
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

## 2. Running now (as of 2026-11-13)

Read at 00:10 PST on 2026-11-13 from the orchestrator's log (`~/.claude/jobs/x35/status.txt`), the process table and
`models/rb_b1c_s5101/latest.txt`.

| where | job | what decision it drives |
|---|---|---|
| GPU 11-11 17:18 to 18:57 (ended) | **The R9 human clone train** (`hclone_r2_s5901`, seed 5901, 45-minute chunks), started by the orchestrator's R9C driver when X69's last fork ended (17:18; X69 is in the landed rows below). It CONVERGED and the clone is registered (`d79f08ec2`). This row is a pointer, and the R9 unit owns its detail. | The clone that the R9 kit's `--human-data init:` / `opp:` take (`d0e2715d8`). |
| GPU 11-12 00:19 to 17:17 PST (ended); its two crosses on the CPU, 17:21 to 23:40 PST | **CE-3d, the trailing-magnet screen: all three forks trained and both crosses are READ (11-12 21:58 and 23:40 PST): PASS** ([PREREG §16](research_state/measurements/ce3d_magnet_screen_2026-11-11/PREREG.md)). Seeds 7321, 7322 and 7323 each ran P\* + 20M at the re-pin (tag `ce3d-pin-9e97cad5`) on the chain `x35-ce3d-nov12`, launcher exit 0, no restart, arm-end K9(b) PASS (`31b0b402c`, `4662ebb53`, `60efef08c`); 12.75 GPU-h. The crosses were played on the agent cores at the play pin (tag `ce3d-pin-333a73dd`) and hold no lane now. This row is a pointer: the result is the first landed row below. | Nothing further: the PASS licenses registering a follow-up only (§3); it moves no recipe and never held BOUNDARY-1. |
| GPU since 2026-11-12 19:26 PST (CE-3d's forks ended 17:17) | **BOUNDARY-1 phase 1 is RUNNING: run [1] `rb_b1c_s5101` was relaunched 2026-11-12 19:26:24 PST, and its early check PASSED at 19:41:28.** The runs go one after another: twelve runs to 50M, about 6 days for phase 1 (§12: ≈ 136 GPU-h and ≈ 10 of starts), under the Training session's driver v4 (`~/gen3ai_archive/b1_phase1_driver_v4.sh`). At the read, `latest.txt` named `checkpoint_25000420_steps.zip`. The first launch (2026-11-12 17:18) was STOPPED at 17:32 by a bug in the look kit's early check, not by anything in the run; the kit is fixed at `1d592e6cd`, and the stopped attempt was discarded whole. Amendment 4 (2026-11-12, blind: no look, no game): the early check and every look are read from the kit at `1d592e6cd`, every run still trains and every game plays at the launch pin `15136b616`; that attempt spent no resume and no spare seed. Amendment 5 (`cf9258ab6`, 2026-11-12, blind, no re-pin): at the launch pin a stop signal that lands inside a compiled update kills the child with no save (fixed on main at `207aa0ae3`, not in the pin; main's launcher reports it as an UNSAVED stop, exit 9); it is no FATAL, VOID or replacement cause, and a run stopped that way resumes on the same pin from `latest.txt`'s checkpoint. §13's GPU items 1 to 7 were GREEN at the launch pin (2026-11-12, 04:54 to 08:51 PST, ≈ 3.95 h of the lease). Measured there: the wide arm costs ρ = 1.392 times the narrow one (734.59 against 527.67 s per 1M steps, pilots to 4M), so the EQ look is at 71,000,000; the width-256 update fits with 1,398.5 MiB of headroom against the 1,024 bar; the 32 CUDA-only slow tests pass; the cross-width cell passes; both pilots read K9(b) PASS. §12 is rewritten from ρ: ≈ 309 GPU-h to the decision (≈ 281 of training, 210 to 281 by which narrow rate holds; ≈ 20 of starts; ≈ 8 of looks), ≈ 15.4 days from launch. Amendment 3 (2026-11-12, blind, before any launch): a startup fit refusal at any start of a run is a start failure, retried from the same checkpoint on the same pin; a repeat on an empty card holds that run. Amendment 2 (2026-11-12) set the base every run types: `--ent-coef 0.0125` (the owner's yes of 2026-11-11 to X67's PASS) and `--behaviour-check warn`, with K9(b) judged by the look. The records are the PREREG's §18 (slots S1 to S7; S8, the per-launch records, is still open, so the relaunch and the early-check time above are from the orchestrator's log). | MF-5 (§3). |
| CPU: the play lanes are FREE | **X60-T look 2 READ 11-11 16:52 PST: NO-TRANSFER** (the landed row below). Its two lanes held the agent cores from 11-09 03:36 to 11-11 16:45; no registered read holds them now. | The next CPU read is the orchestrator's pick from §3 (X67's conditional Kakuna read and the searched player's cost read both waited on these lanes). |

### Landed since 2026-11-01

| when | job | result | what it decided |
|---|---|---|---|
| GPU 11-12 00:19–17:17; CPU crosses read 21:58 and 23:40 PST | **CE-3d, the trailing-magnet screen: PASS** (PREREG `08b998d12`, Amendments 1 to 3, the last `e9f55e723`; full tier). Three P\* + 20M forks with the entropy bonus replaced, at the same weight, by a pull toward a copy of the policy refreshed every 2M steps (`--ent-coef 0 --magnet-kl-coef 0.05`, seeds 7321–7323, tag `ce3d-pin-9e97cad5`) against all eight C0 finals (`ent_coef` 0.05, pin `f4b25238`), head to head with X67's cross and rule, 48,000 games: **D +3.97 pp [+3.31, +4.63]** (90 %; the row floor bound; all 24 cells above 50; the fresh-run-floor reading [+1.35, +6.58]; banked columns +4.14, top-up +3.68). No VOID (every §8 guard executed per fork). Label ENT-LOW: end entropy 0.49 nats against C0's 1.01 and X67's 0.68, still falling at +20M; no fork LR-MOVED (0.0001005). REPORTED, deciding nothing: against X67's three finals (18,000 games) **+1.01 pp [−0.12, +2.13]**; beside X67's own +1.97 pp the swap's D is +2.00 pp higher, with no interval. The magnet's pass costs ≈ 3.5 s a rollout (≈ 4.9 %). No Kakuna game ([PREREG §16](research_state/measurements/ce3d_magnet_screen_2026-11-11/PREREG.md)). | The swap forks beat the unchanged recipe's forks by twice X67's margin, and ended sharper than X67's. The registered reading of the cross against X67's finals: "the swap's gain over C0 is plausibly the lower uniform push alone"; with ENT-LOW, "the swap won and ended sharper; not separable from X67's effect here". C0 trained at another pin on earlier days (CPU-only evidence that this does not matter); siblings are not Kakuna; exploitability and cycling were not measured. | It licenses REGISTERING a follow-up on new seeds with the control that separates the anchor from the missing push (an `--ent-coef 0` arm with no magnet at the CE-3d pin), and nothing else. None is registered. No recipe, no production config and no BOUNDARY-1 arm moves. |
| GPU 11-11 04:43–17:18, read 17:35 PST | **X69, the move-belief LIGHT SCREEN: DROP** (registered `228dec074`, no amendment; scored at `75abf79d8`). Three forks of P\* to +20M steps, one run each, against the five banked C0 finals on the off-pool belief meter's proper move score (`moves_ll`, nats a unit, arm − C0, 72 teams a panel, 95 %). Bar: lower bound ≥ +0.6 on held-out teams (H), > 0 on forum teams (F), ≥ −0.2 on home72. The move head trained by its labels alone (A, `--belief-grad-mode move_label_only`): H **+0.552 [+0.480, +0.624]**, F +0.571 [+0.497, +0.649], home72 +0.531 [+0.463, +0.599]. The move latent loss off (B): +0.046 [+0.001, +0.088] / +0.093 [+0.047, +0.140] / +0.046 [+0.015, +0.077]. Both (AB): +0.494 [+0.427, +0.559] / +0.546 [+0.476, +0.621] / +0.544 [+0.475, +0.616]. No arm VOID; 12.55 GPU-h; by hand equal ([SCREEN §9](research_state/measurements/x69_move_label_only_screen_2026-11-10/SCREEN.md)). | Every arm misses the H bar and meets the other two. The label-only forks close 27 to 30 % of C0's 1.83-nat deficit on H and stay 1.27 and 1.33 nats below the Smogon prior there. Screen evidence, never a verdict: one run an arm, no game played. | MF-3's move-leg gradient lever closes at this dose; both flags stay at their recipe values; the forks play and seed nothing. A bounded residual or a trained-in floor stays open for a fresh run. |
| CPU 11-11 16:52 PST | **X60-T look 2 (final): NO-TRANSFER** (rule `x60`, AM-X60T-2 `aea028c60`; run record and K9(b) `1f3a3b38a`). P\* + 20M with human ladder TEAMS in half the pool (H-TEAMS, s3501–s3508) against C0 (s3501–s3508) on forum72 vs Kakuna: **Δ_F −0.23 pp [−1.95, +1.49]**, NO-USEFUL-GAIN (0.5107 vs 0.5130, 9,600 games an arm; equivalent within ±3 pp). In-house: the positive control R72 (the corpus's own teams) **+1.69 pp [+1.08, +2.30]** DELIVERED; forum72 I_F −0.34 pp [−1.14, +0.46], not a gain; home72 Δ_P **−2.12 pp [−2.99, −1.25]**, inferiority not excluded. Against the forum-team lever D1 (5 v 5, H + F): +0.60 pp [−1.58, +2.78], NO-DIFFERENCE-AT-3PP. Determinism guard PASS; all 21 runs pass the VOID gate; three units lost to Kakuna's long-tail fault were re-played by the tool's resume; COMPLETE ([`read_look2/`](research_state/measurements/x60t_h_teams_2026-11-06/read_look2/README.md)). | Human ladder teams in the pool made the plateau policy better on those teams and no better on human-built teams it had not practised, at a 2 pp in-house cost on the pool's own. Reported only, never decided: P72 +5.80 pp and H +2.77 pp against Kakuna (140 games a final; both panels are adjacent to the corpus). | H-TEAMS stops at the plateau for this six-seen, loser-enriched slice. BOUNDARY-1's §10 HOLD on human teams does not fire (it needed ADOPT-CANDIDATE); no recipe moves. MF-6 (X62) drops below MF-7 (EXPERIMENT_BACKLOG §2.0). |
| CPU 11-11 12:32 PST | **X67, the entropy screen: PASS** (PREREG `1bab9e87a`, unamended; pin `f4b25238`). Three P\* + 20M forks at `ent_coef` 0.0125 (seeds 6701–6703) against all eight C0 finals (0.05), head to head, both sides greedy, 48,000 games: **D +1.97 pp [+1.35, +2.59]** (90 %; both binomial floors bound; the fresh-run-floor reading [−0.65, +4.58]; banked columns +1.95, top-up +2.00). No VOID; every fork LR-MOVED (8.372e-05 against 1.0047e-04, dose rate 0.83×); end entropy 0.68 against 1.01 nats. No Kakuna game ([README](research_state/measurements/x67_entropy_screen_2026-11-09/README.md)). | Names 0.0125 for BOUNDARY-1's base, after the orchestrator weighs the LR label and the owner says yes (§6). It moves no recipe by itself. |
| GPU / CPU 11-10 | **Model-first builds and ops fixes.** (a) **Move-leg lever** (`3c7f3bedd`..`228dec074`): the move-belief diagnosis (the unshown-move belief is 0.6–2.2 nats a unit below its Smogon prior off the pool), `--belief-grad-mode move_label_only` built off by default, the meter's `moves_ll` column (`811c08ddc`), and X69 registered. (b) **CE-3d magnet KL** (`251bbfb87`, v170): `--magnet-kl-coef`, off by default; its screen is read (CE-3d, the first row of this table). (c) **R9 human clone kit** (`d0e2715d8`): `--human-data init:` / `opp:`. (d) **Core-split A/B**: no narrower arm passed, 5 + 3 kept (`1ffe6902b`). (e) **Ops**: one rule for "is a training run live" (`7c25230ba`), two static-gate blind spots (`b27c78383`), the launcher refuses a GPU launch from a shell above nice 10, judged on the resolved device (`6e4a0300e`, `d4b0576ea`), a flaky heavy-slots test (`f85f7f3dd`), and the overlap golden re-record with its numerics-goldens guard (`f819a7271`). | Nothing here is a verdict; every lever is off by default. | Feeds X69 and BOUNDARY-1's checklist; no run changes. |
| GPU / CPU 11-10 18:00 PST | **X66 ExIt-0H: NOT-DELIVERED** (PREREG `a2dbc9616`, Amendment 1 the bank; MF-1). Three P\* students, each fine-tuned toward the HONEST search S′h's piKL targets on two panels' banked decisions (188,297 honest labels; the probe's stronger recipe), agree with S′h's pick on the panel they never saw LESS than P\* does: pooled Δagree **−7.05 pp [−7.31, −6.78]** (H −6.48, F −7.16, P72 −7.50). By hand equal; all three exports bit-identical at the play pin; no VOID. On unseen deals and unseen teams alike Δagree is ≤ 0 at every checkpoint, and 21–24 % of a student's own departures from P\* land on S′h's pick (a gain needs about half). No Kakuna game was played ([`x66_exit0h_2026-11-09/`](research_state/measurements/x66_exit0h_2026-11-09/README.md)). | At this recipe and ≈ 113k honest labels a fold, a P\* student does not learn S′h's choices on unseen teams. | By the registration's pre-commitments: NOT evidence for the scale bet (MF-5), and no ExIt-1 is planned on it. S′-HONEST's game gain is untouched. |
| CPU 11-09 03:00 PST | **S′-HONEST: GO-HONEST, GOAL-B-REACHED** (PREREG `e81f80657`, pin `e46e3ddf8`, no amendment). P\* + S′h (`r0s4_honest_inc@10`: worlds built from what our side has seen alone, every eligible decision from turn 10) against P\* alone, both against the real Kakuna, on 5,097 fresh paired unfamiliar-team deals: **+2.00 pp [+0.34, +3.51]** (worst-case-imputed LB +0.32; reported panels P72 +4.69, H +1.32, F 0.00). The searched arm scores **0.5145 [0.5008, 0.5282]** over 5,100 deals, ties as non-wins (raw 0.4948). Every guard PASS (G-PREFIX 0 mismatches; the identity re-score 600 / 600 at the pin, rebuilding every carry; 3 drops, Kakuna's turn-cap fault). Exclusions 5.6 % (F 10.0 %), above stage 1b's projection, under every bound. 52 h on the whole box ([`sprime_honest_play_2026-11-06/`](research_state/measurements/sprime_honest_play_2026-11-06/README.md)). | "Goal (b) met with search": P\* + S′h beats Kakuna (checkpoint 34, greedy) on U = H, F, P72, in that regime only. **Clustering addendum (11-09, REPORTED; [`addendum_clustering/`](research_state/measurements/sprime_honest_play_2026-11-06/addendum_clustering/README.md)):** the gain holds under every team clustering (LB ≥ +0.31 pp); the score's 0.50 clearance does not ([0.494, 0.535] on our team, [0.491, 0.538] two-way), and the gain is detected on P72 alone. | The search program for goal (b) is un-paused: a cost read and the deployable player's design may be registered. No public play, no training or league change. |
| CPU 11-07 00:47 PST | **X60-T look 1: CONTINUE** (`af57d4fd3`). P\* + 20M with human ladder TEAMS in the pool (H-TEAMS) against C0 on forum72 vs Kakuna: **+0.64 pp [−2.17, +3.45]** (X60-T 0.515 vs C0 0.508, 3,600 games an arm; per seed −0.42 / +0.17 / +2.17). Determinism guard PASS; every VOID gate PASS. | Neither a GAIN nor a NULL at look 1. | Look 2 licensed; registered by AM-X60T-2 (`aea028c60`); its training started 11-09 (§2). |
| GPU / CPU 11-06 05:46 PST | **X65 learnability probe** (DESCRIPTIVE, no verdict; `0b9e6f354`): can a student learn S′'s departures from P\* at a 3–10× stronger recipe? A same-rate control target transfers to unseen teams (+11.70 pp [+11.05, +12.32]) and a capacity arm memorises S′'s departures in-sample, but the curve arms' held-out Δagree is ≤ 0 at every checkpoint (val-selected −4.79 / −5.16 pp) ([`x65_learnability_probe_2026-11-06/`](research_state/measurements/x65_learnability_probe_2026-11-06/README.md)). | Reading 2: S′'s (leak-inclusive) departures are unlearnable at this scale. | ExIt on S′-LIVE's labels stops here; ExIt-0H on S′-HONEST's honest labels would need its own registration. |
| code 11-06 21:08 PST | **The our-HP exact fix** (`83272a8f9`, `gen3_our_hp_exact_v1`, config v165): every HP of our side reaches the damage operator as an exact integer (a Seismic Toss against exactly 100 HP was read as no KO), and heal worlds truncate as Showdown does. On P\*'s 16,785-row population: 6.7 % of rows move at all, 0 top-action changes. | A P\* forward correction, both arms alike. | S′-HONEST was pinned after it; runs pinned earlier keep their op. |
| code 11-06 16:56 PST | **The T2 parity tie-judge fix** (`3260d7ae0`, `gen3_t2_parity_tie_judge_v1`): X60-T s3501's T2 FATAL was a near-tie flip (two Hidden Power types 1.6e-6 apart at a discrete cut, resolved the other way by the compiled kernels' autotune), not a miscompile; every parity gate now judges ties by K9(b)'s one rule. GPU confirm PASS. | Root rule 8 for the parity gates. | X60-T stays on its side pin 7fe15726. |
| CPU 11-06 | **S′-HONEST's incremental builder** (`f7da935ab`, S′H-15 `335355e6f`): `r0s4_honest_inc` carries its worlds through a battle, still a function of what our side has seen (the leak test passes in sequence mode): 6.2 CPU-min a searched game at K = 6 against the cold builder's 30.3 (**4.9× cheaper**), exclusions 3.4 % against 7.7 % offline. | The replication became affordable (5,100 deals, ≈ 2 days of the box). | S′-HONEST registered on it (`e81f80657`). |
| CPU 11-06 01:05 PST | **X65 ExIt-0: NOT-DELIVERED** (PREREG Amendments `df7288a04`, `42dff4e9d`, `ff6874fcc`). Three P\* students, each fine-tuned toward S′'s piKL targets on two panels' banked decisions, agree with S′'s pick on the panel they never saw LESS than P\* does: pooled Δagree **−1.26 pp [−1.40, −1.11]** (H −1.06, F −0.94, p72 −1.78; 97,395 label rows). By hand equal; P\*'s offline forward reproduces its recorded choice on 100 % of rows. No Kakuna game was played ([`x65_exit0_2026-11-05/`](research_state/measurements/x65_exit0_2026-11-05/README.md)). | One supervised distillation pass at this recipe does not deliver S′'s choices; it does not close ExIt (honest labels, larger budgets, in-loop forms). | ExIt-1 is not funded. |
| CPU 11-05 16:47 PST | **S′-LIVE: GO-LEAKY** (PREREG `9c2f023e4`, AM-K6 `d93a51124`). P\* + S′ (`r0s4_rolls@10`, every eligible decision from turn 10) against P\* alone, both against the real Kakuna, on 2,698 fresh paired unfamiliar-team deals: **+4.60 pp [+2.66, +6.48]** (worst-case-imputed LB +2.57; H +4.5, F +5.2, p72 +4.1). The searched arm scores **0.545 [0.526, 0.564]** (raw 0.498). Every guard PASS (G-PREFIX 0 mismatches, identity re-score 600 / 600, 2 post-treatment drops). ≈ 315 CPU-h, 102 h over four supervisor lives ([`sprime_live_play_2026-11-01/`](research_state/measurements/sprime_live_play_2026-11-01/README.md)). | Goal (b) is "REACHED (leaky)" at most: S′ sees the true battle seed. | Licenses ONLY a leak-honest replication on fresh deals; no training, league or public-server change. |
| CPU 11-04 03:14 PST | **N4 look 1**, the toy capacity screen (T\* with every trunk FFN grown 256 → 1,024, +70M, against a same-pin C0′; 3 + 3 seeds; PREREG `8ed8164ea`, AM-N4-FIXES `801a96d0e`, NOTE 2 `6aa46f9e0`): **NO-GO.** In-house **+0.46 pp [−0.28, +1.20]** (90 %; S at 2,000 pairs per cell); Kakuna on forum72 **+0.20 pp [−2.91, +3.31]**, UNDECIDED (N4 0.431 vs C0′ 0.429, 3,000 games each). The new units were used, at about the scale of ordinary 70M drift (registered reading "used, buys nothing"). Every registered check PASS; every row at `31bd9ced9`. Survived a GPU fault (11-01) and a box reboot (11-03): the four remaining arms re-ran under the same names and pin. README `research_state/measurements/n4_capacity_screen_2026-10-27/`. | Closes growth at the plateau on the toy within 70M (depth X47, FFN width N4). | No N4-P on toy evidence; the scale bet loses its cheap support (not refuted). |
| CPU 11-03 11:59 PST | **X59-S's ≥ 1500 follow-up** (in-house only; 15 fresh students at 270,260 rows each; 30 cells of 200 pairs on the read's one engine, played from the pin tag x59s-play-08271d24 (a commit off main) under Amendment 8 because the registered pin could not load the v164 students; `71ef78a79`). **Rating: NOT-RESOLVED**, S-H≥1500 − S-H-mix **+1.33 pp [−2.76, +5.43]**. **Data: SELF-PLAY-DATA-BETTER**, S-H≥1500 − S-S cut **−4.42 pp [−6.07, −2.76]**. Read as rating at a fixed slice and volume: the same panel-excluded, forfeit-heavy slice, with concentrated accounts and shared rows. | Higher-rated human rows still teach a fresh network less than our self-play rows, and the rating effect is unresolved. | Routes nothing alone; human data's closure stays blocked (Δ_restriction), H-TEAMS (X60-T) stays first. |
| CPU 11-01 08:09 PST | **The deviation-only live read: STOP.** At 1,793 departures of the search S′ (`r0s4_rolls@10`) from P\* on unfamiliar teams, S′'s pick was forced in a replay of the game against the real Kakuna. It won **+1.20 pp [−1.55, +3.88]** a departure more than the game as played (H +4.2, F +0.5, p72 −1.1; home72 +2.7 [−4.0, +9.6], reported). The self-referee's own edge on the same fresh departures is +0.99 pp, against +3.4 on the gate's states. Every registered check passes: one engine, 135 / 135 null replays identical, 9 / 2,100 branches dropped. All 9 drops are Kakuna replay divergences, made drops instead of a VOID by the blind AM-DL-PEER (`e40d9663c`, `276c5b7e6`). 36.6 CPU-h. README `measurements/dlive_read_2026-10-28/README.md`. | Search stays paused for goal (b). | No amended S-U on S′. |
| GPU 10-30 14:46-15:19 | **Canary slice-2 diagnosis** (`6b3966dc6`; verdict `7775b2241`): the compiled-numerics hypothesis is **RULED OUT at 5 states**; every state reads CONDITIONING (compiled median distance 0.09-0.55 of the fp32 dither null's maximum). The 30 GPU slow-tier rows all PASS (`1a07fd819`). | The refusals are an ill-conditioned single-row-carried parameter against a lucky eager denominator, not a compiler fault. | The bf16 per-parameter bar now divides by the fp32 dither null (`gen3_canary_param_null_v1`) **at new pins only**; runs pinned earlier keep the eager bar. The earlier stage-2 read stays **INCONCLUSIVE**: the row bar's cap stays 0.72 (`designs/training/canary_recalibration_2026-10-28.md`). |

## 3. The plan, ranked (as revised 2026-11-09: model-first)

The owner's 2026-11-09 direction (§1) puts MODEL levers first on the GPU, ranked by expected strength per GPU-h in
[`research_state/EXPERIMENT_BACKLOG.md`](research_state/EXPERIMENT_BACKLOG.md) §2.0; below it, the orchestrator's
order after REVIEW_2 of the all-null strategy (`69c7cd34d`) with the verdicts since 10-27 applied. Every item needs
its GO.

**The GPU after X60-T's training (ended 11-10): the model-first queue** (EXPERIMENT_BACKLOG §2.0; every effect a GUESS).
- **MF-1, ExIt-0H (X66): READ 2026-11-10, NOT-DELIVERED**
  ([`research_state/measurements/x66_exit0h_2026-11-09/README.md`](research_state/measurements/x66_exit0h_2026-11-09/README.md)):
  P\* students fine-tuned toward S′h's 188k banked HONEST labels agree with S′h on held-out panels less than P\* does
  (pooled Δagree −7.05 pp [−7.31, −6.78]); no Kakuna game was played. By its registration this is not evidence for
  MF-5 and plans no ExIt-1 (MF-1b): any follow-up needs its own case.
- **X67, the entropy screen: READ 2026-11-11, PASS**
  ([`research_state/measurements/x67_entropy_screen_2026-11-09/README.md`](research_state/measurements/x67_entropy_screen_2026-11-09/README.md)):
  three P\* + 20M forks with a quarter of the entropy bonus (`ent_coef` 0.0125) beat the eight banked no-lever forks
  head to head, **D +1.97 pp [+1.35, +2.59]**, all three LR-MOVED (the controller's rate 8.372e-05 against C0's
  1.0047e-04). By its registration that names 0.0125 for the base recipe of every BOUNDARY-1 arm. **The owner said
  yes on 2026-11-11** (in writing, to the orchestrator), and BOUNDARY-1's Amendment 2 (2026-11-12) types it on
  every run. For BOUNDARY-1 only: the production default stays 0.05.
- **CE-3d, the trailing-magnet screen: READ 2026-11-12, PASS**
  ([`research_state/measurements/ce3d_magnet_screen_2026-11-11/PREREG.md`](research_state/measurements/ce3d_magnet_screen_2026-11-11/PREREG.md), §16):
  three P\* + 20M forks with the entropy bonus replaced, at the same weight, by `β · KL(π ‖ π_mag)` toward a copy of
  the policy that is replaced every 2M steps (`--ent-coef 0 --magnet-kl-coef 0.05`) beat the eight banked no-lever
  forks head to head, **D +3.97 pp [+3.31, +4.63]**; no VOID; no fork LR-MOVED. Label ENT-LOW: the forks ended at
  0.49 nats, below C0's 1.01 and X67's 0.68, and were still falling. Two limits were registered and both bear on
  the number: the swap also removes the standing push toward uniform, which X67 showed wins by itself, and C0
  trained at another pin on earlier days. On the first, the REPORTED cross against X67's three finals reads
  **+1.01 pp [−0.12, +2.13]**, and its reading, fixed in advance, is "the swap's gain over C0 is plausibly the lower
  uniform push alone" (that cross misses a true +1 pp three times in four; through C0 the swap leads X67's forks by
  +2.00 pp, no interval). **The PASS licenses registering a follow-up and nothing else**: new seeds, with an
  `--ent-coef 0` arm with no magnet at the CE-3d pin as the separating control. It is NOT registered, and no recipe
  moves.
- **MF-2: BOUNDARY-1**, one retrain boundary for E13 (their max HP as a belief; built, config v168), round 2 of the
  mechanics fixes and E9, each screened first by a forward census on P\*'s rows; trained by MF-5's fresh runs, no GPU
  of its own.
- **MF-3:** the opponent belief off the pool. Its meter is built (`python -m main.offpool_belief`) and the baseline
  REPORTED (ledger 2026-11-09): P\*'s species belief is 3 to 8 nats worse than its Smogon prior on held-out teams,
  team breadth in the loop brings it to the prior with the head unchanged, no arm beats the prior. Decision: no head
  reshape for BOUNDARY-1 (`endstate/design_offpool_belief.md`). Step (b) is screened out: the light screen X68 (P\*'s
  belief residuals retrained on procedural-team labels, trunk frozen) read DROP at `9ba2e80b8`, +3.2 / +2.3 nats
  better than P\* on F / H but still 0.74 / 0.68 below the prior (screen evidence, never a verdict). Breadth in the
  loop is the measured route; BOUNDARY-1's 2 × 2 tests it. The MOVE leg is diagnosed (ledger 2026-11-10): an
  over-confident species × move table held up by gradient that is not the label's; the training lever
  `--belief-grad-mode move_label_only` is built off by default; its light screen X69 (three forks of P\*,
  12.55 GPU-h, `228dec074`) read DROP on 2026-11-11 (§2's landed row): the label-only forks gain +0.49 to +0.55
  nats a unit on H against a bar of +0.6 on the lower bound, so the lever is screened out at this dose (screen
  evidence, never a verdict).
- **MF-4:** the value at the leaf on unfamiliar teams, V-RANK first (CPU).
- **MF-5: BOUNDARY-1's from-scratch 2 × 2**, width (128 against 2× `D_MODEL`, `--d-model`, `0220f8408`, config
  v169) × team breadth (the pool against half procedural teams): REGISTERED 2026-11-09 at pin `507d0c119`,
  **AMENDED 2026-11-10 to three seeds a cell, twelve fresh runs to 100M** (Amendment 1, after an independent
  pre-launch review), LAUNCHED 2026-11-12 (phase 1 running, §2)
  ([`research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md`](research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md));
  the orchestrator adopted the 2 × 2 over the draft's bundle and width-alone options. **AMENDED 2026-11-12
  (Amendment 2, blind):** every run types `--ent-coef 0.0125 --behaviour-check warn` (the owner's yes to X67's
  PASS; the production default untouched), the look judges K9(b) itself (only a PASS lets a run count; a VOID run
  is lost to its lever and is not replaced), and the width-256 memory fit is recorded (it failed the startup fit check by 2.5 MiB of headroom at
  `aee363bc9`, and fits with 1,318.5 MiB against a 1,024 bar since `e112cbcbf`). **The launch pin is `15136b616`**
  (its §18 re-pin record, 2026-11-12; the registered pin launches nothing). **§13's GPU items 1 to 7 are green at
  that pin (2026-11-12; §18, slots S1 to S7):** the wide arm costs ρ = 1.392 times the narrow one (measured on two
  pilots to 4M), the EQ look is at 71M, and the width-256 update fits with 1,398.5 MiB of headroom. **AMENDED
  2026-11-12 (Amendment 3, blind):** a startup fit refusal at any start of a run is a start failure, retried from
  the same checkpoint on the same pin; a repeat on an empty card holds that run. ≈ 309 GPU-h to the decision (≈ 281
  of training, 210–281 by which narrow rate holds; ≈ 20 of starts; ≈ 8 of looks), ≈ 15.4 days from launch; the
  the launch was 2026-11-12, after CE-3d's forks. Width is decided on the in-house cross over all four
  panels,
  breadth on forum72 alone and kept only if home72 pays no more than forum72 gains; Kakuna clustered by team
  confirms either. Honest power: a true +4 pp reads WIN 19 times in 20 if the seed spread at 100M is the 1.76 pp
  measured at 15M, 4 in 5 at 2.5 pp, 5 in 8 at 3.1; a true zero reads "not shown at this size" 93 % of the time and
  UNDECIDED 3 % (27 % at two seeds, which is why the third seed is bought up front and the extension withdrawn); the
  seed spread on forum72 alone is unmeasured.
- X60-T read NO-TRANSFER (11-11, §2), not ADOPT-CANDIDATE: BOUNDARY-1's §10 HOLD on human teams does not fire, and its team recipe stays the pool's.

**Search engineering for goal (b): UN-PAUSED by S′-HONEST's GO-HONEST (2026-11-09), CPU-side and lower priority**
([`endstate/search_ledger.md`](endstate/search_ledger.md) Q19).
- The deployable player is designed ([`endstate/design_searched_player.md`](endstate/design_searched_player.md)) and
  its **cost** read (a budget or builder cheaper than `r0s4_honest_inc@10`, whose searched game costs 6.6 CPU-min) is
  DRAFTED, not registered. Its stage-0 tooling goes on; its stage-1 games (540–765 CPU-h) wait until no model lever's
  read needs the lanes. Public play still needs the owner's approval file and the drift scan.
- Distillation of S′'s leak-inclusive labels stops at this scale (X65 NOT-DELIVERED; the learnability probe's reading
  2); the honest-label rung is MF-1 above.

**X60-T (H-TEAMS): READ 11-11, NO-TRANSFER** (look 2, final: Δ_F −0.23 pp [−1.95, +1.49] on forum72 vs Kakuna; the corpus's own teams +1.69 pp in-house; look 1 CONTINUE, `af57d4fd3`; §2). Human ladder teams in the pool stop at the plateau for this slice of the corpus; with X63 (item 2) this closes the breadth family's plateau members.

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
- **Then S′-LIVE read GO-LEAKY** (2026-11-05): S′ playing every eligible decision against the real Kakuna on 2,698 fresh unfamiliar-team deals, paired with P\* alone, wins +4.60 pp [+2.66, +6.48], leak-inclusive by name (S′'s worlds replay the true seed). **Its leak-honest replication, S′-HONEST, read GO-HONEST** (2026-11-09; [`sprime_honest_play_2026-11-06/`](research_state/measurements/sprime_honest_play_2026-11-06/README.md)): +2.00 pp [+0.34, +3.51], searched U score 0.5145 [0.5008, 0.5282], GOAL-B-REACHED as registered (team-clustered [0.494, 0.535]: not established beyond these panels' teams; the gain is detected on P72 alone). dlive's per-departure STOP stands as written; the three are different estimands, never pooled.

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
| **Kakuna** | Metamon's best public model; 142.8M parameters, offline RL on human and self-play data | home72 **0.598** (ahead); unfamiliar **0.487** (≈ even, goal not met); P\* + S′h on unfamiliar **0.5145 [0.5008, 0.5282]** (ahead as registered, "goal (b) met with search"; team-clustered [0.494, 0.535], not established) | `3010c81ba`, `e81f80657` |
| **PastaMittens** | Metamon's ensemble | U 0.484 [0.456, 0.512], home72 0.550 [0.501, 0.598]: no stronger than Kakuna | `82f273eff` |
| **Metamon human-only models** | trained only on human replays (SmallRL 14M, LargeRL 195M, LargeIL 194M parameters) | P\* beats all three (0.63–0.79). Offline RL beats imitation on the same data by about 16 pp. | `4d8d9eef8`, `df35d1bb0` |
| **Foul Play** (search bot; Gen 9 OU champion) | MCTS over a hand-written evaluator | **By proxy only:** Kakuna draws it, 0.490 [0.422, 0.559], and P\* beats Kakuna on home72. **Never read against P\*.** Our old 75M policy lost 0.388 [0.288, 0.497]. | `8cba100f8`, `e00e78da0` |

So: at the state of the art on familiar teams, with a model about 55 times smaller (70 times by the feature extractor
alone), and level with it on unfamiliar teams without search, an interval that spans 0.50 (the survey words it "behind";
[`research_state/literature/sota_survey_2026-10-26.md`](research_state/literature/sota_survey_2026-10-26.md), `fd3a99b65`).
With the leak-honest search S′h on every eligible decision it scores 0.5145 on unfamiliar teams: ahead as registered
(LB 0.5008, deals treated as independent), not established once clustered on teams (LB 0.494) (S′-HONEST, `e81f80657`).

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

5. **The scale bet: a network wider from step 0** (EXPERIMENT_BACKLOG §2.0 MF-5;
   [`endstate/strategy_if_levers_null_2026-10-26.md`](endstate/strategy_if_levers_null_2026-10-26.md) §5 question 3).
   The ask as first priced: one seed of production width and one of 2× `D_MODEL`, both from scratch at the next retrain
   boundary, ≈ 52–73 GPU-h to a registered 100M look and ≈ 145–205 GPU-h to the plateau, plus a width build (since landed: `--d-model`, `0220f8408`, config v169). **Re-sized
   2026-11-09 (DRAFT):** one seed an arm cannot clear the measured seed-to-seed spread; four seeds an arm can, at
   ≈ 207 GPU-h to the 100M look (≈ 9 days of our card; a second card would halve the wall), as a bundle, as width
   alone, or as both crossed
   ([`research_state/measurements/boundary1_scratch_2026-11-09/DRAFT_DESIGN.md`](research_state/measurements/boundary1_scratch_2026-11-09/DRAFT_DESIGN.md)
   §0). **Registered 2026-11-09** as both crossed at two seeds a cell, the orchestrator's choice
   ([`PREREG.md`](research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md), pin `507d0c119`), and
   **amended 2026-11-10, before any launch, to three seeds a cell**: ≈ 309 GPU-h to the decision (≈ 15.4 days of
   our card; re-priced 2026-11-12 from the measured ρ = 1.392), launched at a new pin taken once, after the `ent_coef` decision. Its cheap support is gone (the N4 toy and X48 read NO-GO, growth at the
   plateau), and it is still the one lever whose point estimate reaches goal (b)'s gap without search. Recommendation:
   yes, staged, launched once the boundary's fixes are in (X60-T's verdict is in: NO-TRANSFER, 11-11, so its HOLD
   does not fire), so the owner's answer is needed before then.
6. **`ent_coef` 0.0125 for BOUNDARY-1: ANSWERED YES, 2026-11-11** (asked 2026-11-10; X67 read PASS 2026-11-11,
   §3). The screen passed, D +1.97 pp [+1.35, +2.59] head to head, with every fork LR-MOVED (rate 8.372e-05
   against C0's 1.0047e-04); the owner said yes in writing to the orchestrator. BOUNDARY-1's Amendment 2
   (2026-11-12) types 0.0125 on every arm, for that experiment only. Nothing is asked here any more.
7. **The quota hook** (2026-11-10). It blocked every session from 10:28 to 12:59 when the Fable weekly window hit
   100 %. A fix scoped to the model that is out of quota is proposed; it waits on the owner's yes.

## 7. The biggest risks and unknowns

1. **No TRAINING lever we have measured closes goal (b)'s ≈ 4 pp alone; only search does, and only as
   registered** (S′-HONEST: +2.00 pp [+0.34, +3.51], at ≈ 6.6 CPU-min a searched game and not yet a deployable player;
   clustered on teams the searched score's LB is 0.494, and the gain shows on P72 alone, H + F +0.66 [−1.39, +2.67]). D1, D3, X52 and the belief mix each read +1 to
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
   yet a best responder still finds +11 pp. Search on the self-referee banks does not pay at P\* (in games against Kakuna leak-honest search does, +2.00 pp), and the plateau trunk has not lost its
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
