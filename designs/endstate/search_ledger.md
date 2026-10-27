# THE SEARCH LEDGER: what we know about search, where it works, where it does not, and how far we are

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from
this doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Next-step design (DRAFT, 2026-10-25, explicit-only, not registered):** [`design_search_next.md`](design_search_next.md),
the owner's goal 5. It makes four points:
- every IMP here is scored by a referee that plays the searched network on BOTH sides, so only the judge has been measured against a
  non-self opponent;
- the leaf's opponent model and V against a non-self opponent were never varied;
- the decision-to-game bridge is unmeasured, and to first order (the performance-difference lemma, ≈ 26 eligible
  decisions a game) +4.6 pp needs only ≈ +0.0018 per decision, against this ledger's ≈ ±0.010 intervals;
- a ranked plan: E2, the bridge and R0's known hole; E1, an opponent-exact referee; E3, human identification and the
  human judge, zero playouts; E4, a live player against Kakuna, gated.

**Proposed read (DRAFT, 2026-10-27, explicit-only, not registered):** [`design_search_u_headroom.md`](design_search_u_headroom.md),
S-U. P\* against P\* + L1's piKL search from turn 10, against Kakuna on home72 and on pooled unfamiliar teams (U), on
one engine, paired by deal. Its co-primaries are the interaction G_U − G_home72 (H-MID's question) and G_U alone (goal
(b); only its negative class, DOES-NOT-CLOSE-B, is creditable, because the builder's leaks inflate a gain). It costs
≈ 12–16 CPU-min a searched game on average (median ≈ 5) on the lean world screen, ≈ 560 CPU-h at 1,200 deals a set
(ESTIMATE from one game, §6.4), so it is a rented-box read. It sees only large interactions (≥ ≈ 7 pp at 77 %). If registered, it replaces E4's registration. **Its live searched player is BUILT
(2026-10-27, §6.4)**; its rule and PREREG are not.
**Its filter (DRAFT, 2026-10-27, not registered):** [`design_v_rank_on_u.md`](design_v_rank_on_u.md), V-RANK. It
measures the within-node ranking S-U assumes: every legal action at a position branched as a real P\* vs Kakuna game
(deterministic replay, so the truth has no opponent model), Q̂'s slope on U against home72. A null or worse slope on U
turns S-U OFF; an intact one only makes it eligible (on P\*'s own states the slope is 0.54 while search gains nothing).
≈ 96 CPU-h central.

**Status (2026-10-26, latest): H-MID READ — P\*'s critic does not see its unfamiliar-team conversion deficit, but it
still ORDERS unfamiliar-team positions as well as home ones** (`../research_state/measurements/h_mid_value_test_2026-10-26/`,
registered `2dd52278`; not a search read, no playout; it bears on search on V). From level positions at turn 10 of X51's
P\* vs Kakuna games:
- P\* wins 0.609 on home72 and 0.501 on U. Its critic V rates the same positions 0.557 and 0.525: **CRITIC-MISCALIBRATED-ON-U**,
  Δbias +0.076 [+0.016, +0.139]. At equal V, P\* wins 8.6 pp [+2.3, +15.1] less on U.
- **V's ordering is intact:** AUC for the outcome 0.667 on U, 0.663 on home72 (0.711 vs 0.712 over all turn-10 states).
  What fails on U is V's LEVEL, which is already there at turn 1.
- **For search (H-MID FINDING 1, contested):** a one-ply search compares successors within one node, so a between-set
  level error in V does not by itself rule out headroom on U. But AUC ranks GAMES, not the ACTIONS of a node, and V's
  action ranking on U has never been measured. The direct test is S-U (the proposed read above).
- **No search read so far had P\* PILOTING an unfamiliar team.** The off-pool bank's unfamiliar teams were the
  opponent's; every other bank is pool teams on both sides.

**Status (2026-10-25): L1's REGISTERED REPLICATION on the plateau bank is READ — the one PAYS-SCOPED cell
REPLICATES, thinly, and it is still the leaf's signal at the 25M policy, not an anchor gain and not P\*'s**
(`../research_state/measurements/search_l1_plateau_2026-10-25/`; the cell and bar written down from L1 §4 before the
read, `1f980728a`; Tier-1 fit frozen; scoring only; **SELF-REFEREE**). Plateau bank, 719 states × the 25M and P\*
anchors:
- piKL (λ 0.015) at the 4-turn leaf: **+0.0097 [+0.0025, +0.0169]; 98.33 % lower +0.0010 > +0.0005: REPLICATES.**
  The anchor's own part is +0.0025 [−0.0039, +0.0089] (not clear); the plain argmax already reads +0.0072 [−0.0025,
  +0.0170]. Gumbel's cell is clear too (+0.0095, lower +0.0006); THRESH's is not. No rule is clear at both primary
  scorers (LEAF-K8 +0.005 to +0.006 at most, not clear): single-bank label PAYS-SCOPED.
- **The gain is the 25M anchor's:** +0.0193 [+0.0089, +0.0303] at 25M, **+0.0001 [−0.0097, +0.0101] at P\***
  (descriptive, post hoc). E2's cell was also on immature 15M policies, so the 4-turn leaf's small positive is a
  property of a less mature policy and is gone at the plateau.
- Nothing is adopted: an anchored rule stays a guard against the argmax's harm (here the argmax does not hurt), never a
  source of gain. The bot stratum is SELF-REFEREE (E1-0: UNDECIDED) and not clear alone (+0.0056 [−0.0060, +0.0178]).

**Status (2026-10-25): the MATURITY BATTERY's PLATEAU STEP is READ — maturity to the plateau does not bring
honest search closer** (`../research_state/measurements/hidden_info_maturity_x26_2026-10-18/README.md` §7; amendment
A1, registered before any of its outputs; GIGO audit 531 / 531; SELF-REFEREE). The record, the plateau bank's
[25M, P\*] on 719 states drawn from P\*'s own games:
- G_F (oracle − honest, LEAF-K8) +0.061 → +0.058; **GAP-SHRINKS NOT DETECTED** (−0.003 [−0.020, +0.015]).
  BELIEF-RISES NOT DETECTED (TODAY 0.001 → 0.014). **FRONTIER-STATIC** (no gate opens at either anchor).
  STILL-MOVING NOT DETECTED; CONVERGED NOT READ (as registered).
- The replication (the interim bank's FULL pair, 723 states) reads GAP-SHRINKS (+0.078 → +0.048, −0.030 [−0.047,
  −0.012]). Not BANK-DEPENDENT by the registered rule (only opposite directions are), so the record stands.
- Honest search stays at or below the policy at every anchor; the oracle's edge falls. R3 at P\* is **SHARED** on
  both banks (E2's 15M: U-FIRST).
- Every cell is a self-referee read; E1-0 (below) shows the bot-stratum gap shrinks against the real bot. A
  bot-referee re-score of the plateau step is PROPOSED (README §7.6), not registered.

**Status (2026-10-25): E1-0 READ — the REAL BOT as the referee on banked bot-stratum picks: UNDECIDED**
(`../research_state/measurements/search_e1_0_2026-10-26/`; registered in its `PREREGISTRATION.md` before any bot
playout; every VOID guard passed: the self-referee replays the banked truth rows exactly, 0 of 28,392 characters
differ; 53 / 53 published bot-stratum points reproduced). The picks the maturity battery, stage 1 and stage 1c banked
on the 160 bot-stratum states (X26 225M) were re-scored with the state's own scripted bot playing the opponent
in-core (`main.value_search.playout.referee`, `opp_bot`), beside the self-referee:
- Primary (98.75 %), D = bot − self: oracle O +0.038 → +0.014, D −0.024 [−0.054, +0.009]; honest F −0.010 → +0.010,
  D +0.020 [−0.010, +0.054]; POOL − F_legal +0.027 → −0.006, D −0.033 [−0.068, +0.002]; stage 1c's C128L piKL
  +0.009 → +0.006. No cell disagrees, none agrees within ±0.015: **UNDECIDED, no relabel.**
- DESCRIPTIVE: every bot-stratum IMP below is a SELF-REFEREE read. Against the real bot the gains shrink (the oracle
  gap G_F on bots +0.050 → +0.010, D −0.040 [−0.068, −0.015] at 95 %), honest arms' slightly negative IMPs turn
  slightly positive (22 secondary cells, D +0.02 to +0.05), and the referees' action rankings agree about half as
  well as each agrees with itself (Spearman 0.19 vs ≈ 0.43, post hoc).
- Consequence: D-1 (`design_search_next.md`) stands and is now measured as a direction, not a size. E1's instrument
  term needs ≈ 4–5× this bank (≈ 650–750 bot states) to resolve ±0.015.

**Status (2026-10-24): R-b2 READ — a HUMAN OPPONENT MODEL as the judge: HOM-NONE**
(`../research_state/measurements/x50_r1b_hom_2026-10-23/`; registered in `design_x50_rung1b_human_opponent_model.md`
§8.4, `42f1d9a7`). The HOM (X26's network fine-tuned from P\* on 1.76M human choices; held-out human NLL 0.58 nats
below P\*'s, R-b1 LEARNED) re-scored the frozen off-pool bank as the opponent-choice judge; P\*'s re-score reproduced
the bank bit for bit (G-REPRO, 320 / 320).
- **Bots: AUC 0.459 [0.406, 0.510], NO-EVIDENCE** (P\*: 0.472); paired ΔAUC −0.013 [−0.068, +0.045], not BETTER.
- Snapshots: 0.706 [0.647, 0.760] WORKS, below P\*'s 0.794 (ΔAUC −0.088 [−0.153, −0.027]); the HOM is no longer our
  network. Pooled 0.583 WEAK (P\* 0.633).
- **Consequence (design §6): the human-opponent-model lever is CLOSED for search at P\*.** Public-team proposals remain;
  search stays paused. Caveat (F-1b-3): the bots are scripted, not human, so R-b2 tests "a better model of humans" as
  a judge of bots; the ladder's own version (L3-H) is unbuilt.

**Status (2026-10-21): STAGE 1c is READ: NEARER BUT NOT PAYING**
(`../research_state/measurements/search_1c_2026-10-20/`; pre-registered `c9bdc928`, AMENDMENT A `a144482b`; PROVISIONAL-POOL).
L3's consequence, tested end to end on the 287 stage-1 states (285 read): 128 honest F worlds and 128 team-level corpus
worlds (C_ladder) per state, each weighted by its damage-roll weight × the opponent-choice likelihood.
- **Phase A, nearness (zero playouts):** all three weighted sets are NEARER the truth than today's 8 F worlds (ΔH, the
  share of the truth's binding hidden facts held: +0.007 F, +0.019 corpus STRICT, +0.013 both). None is MATERIAL, and
  all are SELF-PLAY-ONLY (bots NOT-DETECTED).
  - The corpus brings the never-seen species by itself.
  - The likelihood's own selection is +0.008 among F worlds and not detected among corpus worlds.
  - The exact-world mass stays ≈ 0 (≤ 0.0003).
- **Phase B, playouts** (L1's fitted piKL anchor, the true world's referee): every played arm is NOT-DETECTED against
  the policy, and Holm rejects none. The best is C128L, +0.007 [−0.003, +0.017] at LEAF-K8 (n 285). None beats today's
  8 F worlds.
- **With stage 1 and L3's off-pool read below:**
  - Exactness carries the value at this leaf, and 1c's worlds stay ≈ 0 % exact.
  - The likelihood's evidence exists only when the opponent is our own network.
  - So the opponent-choice route, with our network as the opponent model, buys no search gain here. What binds is
    still a proposal set that holds the truth's TEAM (POOL's identification), which an open population does not give.

**Status (2026-10-21): the world sampler's STAGE 1 is READ — only TEAM IDENTIFICATION pays, partly right worlds do not** (`design_world_sampler.md` §6.2; `../research_state/measurements/world_sampler_stage1_2026-10-19/`; registered, amendments A1–A3 before any outcome was read; PROVISIONAL: pool opponents, X26 225M). On all 287 maturity LEAN states:
- **POOL** (the training pool's consistent teams) beats today's honest worlds (F) by **+0.038** [+0.022, +0.055] per decision at LEAF-K8. It recovers **0.72** [0.53, 0.92] of the oracle gap (PAYS, MATERIAL), on bots and own snapshots alike under the self-referee; against the real bot (E1-0, DESCRIPTIVE) its bot-stratum Δ reads −0.006 [−0.034, +0.019] where the self-referee read +0.027 (UNDECIDED).
- Every **open-population** sampler is NOT DETECTED against F, with points −0.008 … +0.002 and intervals ≈ ±0.015: joint Smogon teammates, a ladder-corpus team model, the forum corpus without near copies of the truth, and the network's species × the forum's sets.
- The network-species arm holds the true never-seen species set in 44 % of its worlds and makes the truth's facts far likelier, yet gains −0.004. Exactness, not calibration, carries the value at this leaf (L2: EXACT-ONLY between arms with POOL, NEITHER without).
- No arm is SEARCH-ENABLED. The best open arm's TODAY is 0.01 (upper bound 0.19), against p\* ≈ 0.2–0.3.
- F2's legality rule (+0.005 [−0.001, +0.011]) and F1 tier A's dice, uniform or likelihood-weighted, move IMP undetectably.

**Status (2026-10-20): L3 REPLICATED OFF-POOL AT THE PLATEAU, SELF-PLAY-ONLY; the pool lift was MEMORY**
(`../research_state/measurements/search_offpool_bank_2026-10-20/`; pre-registered `7642df0f6`, bank `5281bfae0`).
X26's P\* (280M) played its plateau roster, every opponent on a ladder team X26 never trained against (`heldout72_hpfix`;
0 exact and 0 ≥ 4-set overlaps with every pool file at its pin). On 318 states, with P\* as the opponent model:
- **Truth vs honest worlds** (L3's metric): pooled 0.633 [0.591, 0.675] WORKS. **Snapshots 0.794 [0.753, 0.834]**
  (L3: 0.796). **Bots 0.472 [0.410, 0.531], NO-EVIDENCE** (L3 on the pool bank: 0.588). By A4 that is
  **SELF-PLAY-ONLY**.
- **An off-pool ladder decoy gets NO lift over honest worlds:** 0.503 [0.462, 0.543], against the pool decoy's 0.575.
  So the counter-test's pool lift was pool MEMORY, not real-team coherence. The truth beats these decoys (0.611 [0.543,
  0.680]; TRUTH-SPECIFIC), carried by the snapshots (0.729; bots 0.504).
- **Caveat:** the bot stratum is 96 % P\* wins (L3's bank: 62 %). Reweighted to L3's outcome mix the pooled AUC is
  0.703, so bots in P\*'s losses are not measured.
- **For stage 1c (its phase A, below):** its NEARER is SELF-PLAY-ONLY too, and off-pool the bots carry no likelihood
  evidence, so a 1c gain can only be a self-play gain until a non-self opponent model exists.

**Status (2026-10-20): STAGE 1c's PHASE A is READ: the opponent's choices pick worlds that are NEARER the
truth, but only a little, only against our own network, and nowhere near the exact world search needs**
(`../research_state/measurements/search_1c_2026-10-20/`; pre-registered `c9bdc928`, AMENDMENT A `a144482b`; zero
playouts).

What ran: on the 287 stage-1 states (285 read), 128 honest F worlds and 128 team-level corpus worlds (C_ladder) per state,
each weighted by its damage-roll weight × the opponent-choice likelihood (L3's).
- **The primaries, against today's 8 F worlds** (ΔH: the change in the share of the truth's binding hidden facts the
  worlds hold, each never-seen species and each seen mon's unrevealed move): all three read **NEARER** by the registered
  rule, none MATERIAL, and all **PROVISIONAL-POOL** and **SELF-PLAY-ONLY** (bot stratum NOT-DETECTED, snapshot NEARER).

  | arm | ΔH | the share it reaches |
  |---|---|---|
  | the 128 F worlds weighted (F128L) | +0.007 [+0.001, +0.014] | 0.442 → 0.450 |
  | the corpus worlds weighted, STRICT (C128L) | +0.019 [+0.012, +0.027] | — |
  | both (M256L) | +0.013 [+0.007, +0.019] | — |
- **Where the gain comes from:**
  - The corpus moves the never-seen species by itself: 8 uniform corpus worlds already hold +0.095 more of them.
  - The likelihood's own selection is +0.008 on the F proposals (snapshot-only) and not detected on the corpus ones.
  - Against bots the likelihood pulls the corpus worlds AWAY from the truth (−0.008).
- **The exact-world mass** (every never-seen species and every seen mon's move set right) stays ≈ 0 in every arm
  (≤ 0.0003), against the ≈ 0.2 the dose curve asks for. 128 proposals buy nothing over 64.
- **The bar is met:** phase B plays the worlds with L1's anchored rule (RUNNING; stop 2026-10-21 21:00 PDT). §1 and §7
  Q15 carry it.

**Status (2026-10-20): L3 of the review is READ: the opponent's own choices DO point at the true world**
(`../research_state/measurements/search_l3_2026-10-20/`; pre-registered `4d149799`; forward passes only, no playout).
X26's 225M network, seated as the opponent, scores how likely the opponent's actual past choices were in each world.
On the 283 stage-1 states the TRUE world beats a random honest world with AUC **0.692** [0.655, 0.728]: **WORKS** by the
registered rule.
- **By opponent:** 0.796 against X26's own snapshots, 0.588 against scripted bots (WEAK).
- **It is evidence, not familiarity:** a control that ignores the chosen action reads 0.481, at chance.
- **The truth's weight is a counterfactual:** if it were inserted among the 8 honest worlds it would take 0.249 of
  the weight (0.360 against snapshots; uniform 0.111). That is NOT the dose curve's p (§5.1): no honest proposal set
  holds the truth, and the figure shrinks as proposals are added. It does not show search would get 0.2–0.35 (review
  F3).
- **Carried by the snapshot stratum; now SELF-PLAY-ONLY (the off-pool replication above):** the pooled WORKS is the average of snapshots (0.796, greedy
  copies of our own network) and bots (0.588, WEAK on its own). It was read at one pre-plateau anchor on pool
  opponents only (review F1, F2).
- **Part of it is pool-ness (the familiarity counter-test, 2026-10-20, NOT-DETECTED):** a pool DECOY that is not the
  truth beats honest worlds clearly (0.575 [0.517, 0.631]); the truth beats its decoys only inconclusively (0.578
  [0.489, 0.664]; 82 states). Against bots the decoys match the truth.
- **The limit:** re-weighting today's honest worlds moves almost nothing toward the truth's hidden moves (+0.005
  [−0.004, +0.013] pooled; +0.014 against snapshots), because they hold no near-true world to find.

**The likelihood is a good JUDGE; the proposals are the binding piece.** Stage 1c (a larger or team-level proposal
set, weighted by it) is now worth building. §1, §7 Q15 carry it.

**Status (2026-10-20): L1 and L5 of the review are READ** (`../research_state/measurements/search_l1_l5_2026-10-20/`;
pre-registered `d443c098`; zero playouts, banked rows). **L1: an acting rule ANCHORED to the policy stops honest search
from hurting but does not make it pay.** Three anchors (piKL, Gumbel's scale, a deviation threshold), each fitted on
Tier 1 and read held out: on X26 they lift the argmax's −0.012 to about the policy (−0.007 to +0.000), never above it
(even a value chosen on X26 itself reaches only +0.004). The registered verdict is PAYS-SCOPED, on one cell (piKL on E2 at
the 4-turn leaf, +0.012 [98.33 %: +0.001, +0.023]), and that cell is the leaf's own E2 signal: the anchor adds +0.002
[−0.005, +0.008] to the plain argmax there. The acting rule is not what binds; the worlds are. **L5 (descriptive):** leaf
correlation 0.97 at the 35 pre-terminal states, disambiguation 0.41 per own turn (0.70 in turns 1–7), near Skat on Long et
al.'s map, so the classic PIMC failures do not explain the null either. **The registered replication on the plateau bank
was read 2026-10-25 (status above): the cell replicates thinly (+0.0097, lower +0.0010), at the 25M anchor only.** §1, N5 and §7 Q13 carry it.

**Status (2026-10-20): a LITERATURE REVIEW of search under hidden information has been written**
(`../research_state/literature/hidden_info_search_review_2026-10-20.md`; no measurement). Its summary and its ranked
experiments (L1–L6) replace §1's open list. Two of its findings qualify this ledger: the "p ≈ 0.2–0.3" of §5.1 is read
as the cost of a miscalibrated prior, not a requirement (untested), and N5's "the prior hurts" holds in the ORACLE world
only.

**Status (2026-10-19): the world sampler's STAGE 1 is REGISTERED and RUNNING** (`design_world_sampler.md` §6.2; `measurements/world_sampler_stage1_2026-10-19/PREREGISTRATION.md`). On the maturity battery's 287 LEAN states at X26 225M, five candidate samplers (joint Smogon teammates, a team-level ladder-corpus model, the forum corpus minus near copies of the truth, the training-pool ceiling, the network's species × the forum's sets) are played against F at HEAD, with PAYS / HURTS / NULL / MATERIAL / SEARCH-ENABLED rules and a registered stop 24 h after launch. F2's legality rule alone (F_legal − F_off) and tier A's rolls (uniform and likelihood-weighted) are read as secondaries. No number yet (§7 Q12). Search can now average its worlds by their likelihood weights (`SearchConfig(weights="likelihood")`, default uniform).

**Status (2026-10-19): F1 TIER A BUILT, default OFF** (`design_world_sampler.md` §3.3.1, §7 F1; builds only, descriptive). The honest builder's true-dice replay made a world reproduce every printed damage number with the truth's own roll. With `WorldSampler(dice="rolls")` any roll that prints what the searched side saw will do, the dice stream kept aligned, and each world carries a likelihood weight. Measured first: today's screen rejects 9.0 % of its candidates that fit on another roll (2.7× the acceptances, 37 of 40 random states). On 60 rebuilt states the exclusions fall 20 → 6 (random 3 / 40 → 0 / 40), the build's cost halves, and the seen mons' facts barely move. Search does not read the weights yet, the default stays `seed`, and the crit roll and the other dice are still the truth's (tier B, not built).

**Status (2026-10-19): the WORLD SAMPLER has its own design, and its zero-playout screen has been READ** (`design_world_sampler.md`; stage 0, post hoc, descriptive; it reads only the banks' inputs, never a playout outcome). On 1,395 states of the maturity interim LEAN and E2 banks (training-pool opponents):
- The network's belief already puts **0.31** of its mass on the exact set of never-seen species at X26 225M (0.21 at 25M), against 0.020 for today's Smogon draws (F). Its value stayed at 0.01 over the same anchors.
- No candidate gets EVERY seen mon's hidden moves right more than 3.5 % of the time (F 1.1 %, the network 2.4 %). So the mass on a world exact on both is ≤ 0.0025 for every candidate, against the ≈ 0.2 search needs. **The sets, not the species, are what the sampler lacks.**
- **5.2 % of the stored honest worlds are ILLEGAL** by Showdown's validator (32 % of states hold one): independent per-slot move draws made impossible egg / event combinations. **FIXED 2026-10-19 (F2):** the builder now redraws every set until the validator accepts it (0 illegal worlds on 316 rebuilt states, no new exclusion); every earlier honest read used the old builder, and worlds before / after are not comparable (§6.1).

Stage 1 (with playouts, ≈ 12–24 CPU-h) is designed with a decision rule, not run. §1, §3 N15, §6.1 and §7 Q11 / Q12 carry it. **Status (2026-10-19, READ 2026-10-25): the maturity battery's PLATEAU STEP was REGISTERED and RUN** (`measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md` §9, README §6–§7): P\* = 280M declared, a PLATEAU bank of 800 states from P\*'s cycles, P\* FULL + O−M / O−U and 25M FULL on both banks. `QUEUE_DONE` 2026-10-23, nothing parked; the read is the plateau-step status line (2026-10-25). **Status (2026-10-19): the MATURITY BATTERY's first LEAN trajectory has been READ** (`design_hidden_info_binding.md` §12.13.7; `measurements/hidden_info_maturity_x26_2026-10-18/README.md` §5; PROVISIONAL, interim bank). On the same 287 states at X26's 25M, 100M, 175M and 225M checkpoints, the oracle − honest gap G_F falls +0.075 → +0.062 → +0.055 → +0.053, but GAP-SHRINKS is NOT DETECTED (−0.022 [−0.049, +0.004]). The network's belief does not rise (TODAY ≈ 0.01 at every anchor; BELIEF-RISES NOT DETECTED), the series is NOT CONVERGED, and the FRONTIER is NOT READ in LEAN (every bucket's gate CLOSED at every anchor). Post hoc: the decline is the ORACLE's edge over the policy falling, not honest search rising (honest search stays at or below the policy); it is flat at the 4-turn leaf. Maturity alone is not bringing search closer. §1, §5.1 and §7 Q4 carry it. **Status (2026-10-18): the MATURITY BATTERY is REGISTERED on X26** (`design_hidden_info_binding.md` §12.13.7; `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md`). An INTERIM frozen bank of 800 states (LEAN 320 nested; half from games against our own snapshots, as E2 recommends) is drawn from X26's newest cycles, because X26 has no plateau yet. The plateau bank, when it exists, supplements it and becomes the verdict of record. LEAN reads at 25M, 100M, 175M and 225M ask whether honest search's gap to the oracle shrinks and whether the network's belief rises as the policy matures. No number yet (§7 Q4). **Status (2026-10-18): E2, the COMPETENT-OPPONENT bank, has been READ** (`design_hidden_info_binding.md` §16.12;
`measurements/hidden_info_e2_competent_2026-10-16/`, pre-registered `5881ee83`). On 1,200 fresh states from games against the
network's own snapshots, R3 reads **U-FIRST** by the registered pooled LEAF-K8 primary: the never-seen Pokémon carry more than
the seen mons' moves on competent play too. It is SCORER-DEPENDENT (tie-free: SHARED) and SHARED on each checkpoint alone. The
moves' hide-only share rises to 0.58 (B4 holds), so they are a NECESSARY share. PROVISIONAL at 15M. §1, §5.2 and §7 Q2 carry
it. **Status (2026-10-15): the hidden-info study's TIER 1 has been READ on the FULLY honest base** (`design_hidden_info_binding.md`
§16.8; registered rules, frozen blinded table, computed not replicated): §1, §2–§5 and §7 now carry its numbers; the pilot's
descriptive stage reads stay in their owners. **Status (2026-10-15, later): the BLIND REPLICA has been COMPARED by a third
party** (`design_hidden_info_binding.md` §16.9; `measurements/hidden_info_tier1_2026-10-13/replica_comparison.md`). On the 978
states where both implementations built every shared arm, G and the moves' minority shares REPLICATE and so does the post-hoc
tie-break step. R3's never-seen half, the stage, recover, eval and dose / placebo rules are NOT replicated: the replica ran no such
arms. §1, §5.2, §6.1 and §7 Q2 carry this. **Status (2026-10-15, latest): the RED TEAM has attacked Tier 1**
(`design_hidden_info_binding.md` §16.10; `measurements/hidden_info_tier1_2026-10-13/red_team.md`; post hoc, no registered verdict
changed): "the moves are not THE binding constraint" SURVIVES; the never-seen class's dominance is WEAKENED (not detected on the
states from games against our own snapshots, nor on the end-state checkpoint alone; equal per hidden fact); "the dose needs
0.2–0.35" is the 8-rollout leaf's quantisation and "FN ≈ Smogon" is false for species (both BROKEN). §1, W9, §5.1, §5.2 and §7
carry it. **Status (2026-10-15, last): the KNOCK-IN FORKS have been READ** (`design_hidden_info_binding.md` §8.7;
`measurements/hidden_info_forks_2026-10-14/`): told the true moves, items or never-seen species for 3M more steps, the network is
no stronger than its control (NULL inside ±2 pp; spreads NOT DETECTED), and withheld at play the moves and species forks are
≈ 2.4–2.6 pp worse (descriptive). Information is a SEARCH lever, not a lever for the amortized policy at this maturity. §1, N8,
§6.3 and §7 Q3 / Q8 carry it. **Status (2026-10-16): the BLIND U-ARM BUILD (red team E1) has been COMPARED by a third party**
(`design_hidden_info_binding.md` §16.11; `measurements/hidden_info_u_replica_2026-10-15/third_party_comparison.md`). Under its
pre-registration it AGREES with the original on the 978 common states at LEAF-K8: s_U 0.32 / 0.31, n_U 0.83 / 0.92, and R3
holds in both. So R3's never-seen half is REPLICATED, scoped to the 8-rollout leaf and the registered bot-heavy mix. In BOTH
builds it fails on the snapshot-game states and on end2 alone, and it flips at LEAF-K4 / K1. §1, §5.2, §6.1 and §7 Q2 carry
it. **Status (2026-10-12): ESTABLISHED at the owner's request** ("try and build a ledger of what we know about search, what
we know about where it's working, where it's not working, how much we need to improve for it to work … establish the
ledger now"). This is an INDEX-AND-VERDICT doc: it holds no measurement of its own. Every row points at the ledger
entry, design section or measurement folder that owns the number, and the owner wins any disagreement
(`research_state/ledger.md` wins over everything; `design_hidden_info_binding.md` and `design_value_discrimination.md`
own the pilot's detail). The BLIND REPLICA's results (`measurements/hidden_info_replica_2026-10-11/`) enter this doc only
through the third party's comparison (§16.9 of the study).

**Scope.** "Search" here is what a player does at a decision: try each legal action, estimate how good the position
after it is (the critic, or rollouts, or a short rollout then the critic), and play the best. It is OFFLINE / evaluation
only: nothing in `src/main/value_search/` is wired into training or play (owner's one-ply scope rule, 2026-09-30;
playout-to-terminal in-loop is deferred). The currency is **IMP**, DECISION IMPROVEMENT: the win probability gained per
decision by playing the scorer's pick instead of the policy's own top choice, judged on held-out truth seeds (§6).

**Evidence tags.** **MEASURED** = a registered or banked read with its interval. **PROVISIONAL** = measured at 15M steps on
two policies (`rb_es_end2_s2001`, the adopted end state at `62085750`; `rb_es_prod_s2001`, production at `95d014fa`) that
were NOT plateaued (each still moved ~0.1 nats and flipped 15–22 % of its top actions per 1M), so it may move at
the plateau (§7, maturity battery). **UNVERIFIED** = a reading with no read behind it. **DESCRIPTIVE / POST HOC** =
read after the data existed; a hint, never a verdict. **PRE-ERA** = measured before the Rustboro era (history, not a
comparator; the current-era re-measure is named beside it). Intervals are 95 % unless stated.

---

## 1. Where search stands today (one screen)

- **The information ceiling is large and the critic cannot reach it.** Playing every action out in the TRUE battle (the
  ORACLE world) improves the policy's pick by **+0.081 (end-state) / +0.071 (production)** win probability per decision
  at 8 rollouts, against the critic's +0.026 / +0.012. PROVISIONAL.
- **FULLY honest search does not beat the critic, nor the policy.** With every hidden fact drawn from Smogon usage
  (the seen mons' items, abilities and spreads included) and kept only if it replays everything our side saw, an
  8-rollout leaf gains **+0.006** [−0.004, +0.017] (critic search on the same worlds +0.000); the whole oracle edge is
  hidden information about the opponent's TEAM (G = oracle − honest **+0.070** [+0.060, +0.081], pooled). MEASURED
  (Tier 1, registered), PROVISIONAL at 15M. **REPLICATED** by the blind re-implementation on the same 978 states
  (+0.0696 vs +0.0715; paired difference −0.002 [−0.013, +0.009]; study §16.9).
- **The NEVER-SEEN Pokémon carry the gap, not the seen mons' hidden moves.** Revealing only the never-seen closes 0.31 of
  G, hiding only them opens 0.82; for the seen mons' moves 0.13 and 0.42. By the registered rule (R3) "the moves are THE
  binding constraint" is **REFUTED**; the moves are a real minority share (a confident WRONG move set costs −0.013; the
  move dose +0.011 at 0.9). Items, abilities and spreads together add ≈ 0.003. MEASURED. The replica reproduces the MOVES'
  shares (s_M 0.11 / 0.12, n_M 0.39 / 0.47 on the same states). It ran NO never-seen arm. **The BLIND U-ARM BUILD (E1,
  §16.11, 2026-10-16) now REPLICATES U's dominance**, at LEAF-K8 on the 978 common states, by the pre-registered criterion:
  s_U 0.32 / 0.31, n_U 0.83 / 0.92, R3's numerators +0.013 / +0.021 and +0.031 / +0.024, and R3 holds in both. Its scope:
  the 8-rollout leaf (at LEAF-K4 / K1 R3 flips, in opposite directions) and the registered bot-heavy mix. With the learnset
  fixed, the moves' hide-only share reads 0.54 in the blind build against 0.38 (the M side still differs, §6.1).
  **RED TEAM (§16.10 of the study, post hoc): WEAKENED.**
  - **What survives.** R3 survives tie-free scoring (stronger), the exclusions (its hide-only side needs no honest base),
    the mis-keyed records and a never-seen leak check.
  - **Where it is not detected.** On the 42 % of states from games against our OWN snapshots: reveal-only U − M −0.13
    [−0.40, +0.13], hide-only +0.14 [−0.10, +0.39], n_M 0.58. The bot − snapshot difference is clear on both sides. It is
    also not met on the end-state checkpoint alone. **Both REPLICATE in the blind U-arm build (§16.11):**
    - R3 fails on the snapshot-game states (blind n_M 0.85 there) and on end2 in both builds.
    - The bot − snapshot difference is clear on both sides in both builds.
  - **Per hidden fact** the two classes are worth the same (1.21 [0.98, 1.58]).
  - **Red-team confidence** that the never-seen class is the largest: ≈ 70 % on the registered mix, ≈ 35 % where play is
    competent. **The third party, after E1:** ≈ 85 % on the registered mix at LEAF-K8; ≈ 30 % where play is competent.
  **E2, the competent-opponent bank (§16.12, 2026-10-18; pre-registered; PROVISIONAL): U-FIRST.**
  - **The bank:** 1,200 fresh states from games against our own snapshots only.
  - **Revealing:** only the never-seen mons beats revealing only the moves by **+0.0169** [+0.0065, +0.0268] (MATERIAL).
  - **Hiding:** only them costs **+0.0114** [+0.0010, +0.0220] more.
  - **Its scope:** SCORER-DEPENDENT. Tie-free, the hide-only side is not clear (lower −0.0002), and each checkpoint alone
    reads SHARED.
  - **The red team's population attack splits:**
    - its reveal-only half does NOT replicate (E2 − Tier 1's snapshot-game states +0.021 [+0.002, +0.038]);
    - its hide-only half DOES (E2 − the bot-game states −0.028 [−0.044, −0.013]).
  - **The moves' shares:** hide-only 0.58 (B4 holds), reveal-only 0.16 (B3 fails).
  - **Confidence** that the never-seen class is the largest where play is competent, at LEAF-K8: ≈ 65 % (was ≈ 30 %).
- **It falls as the game goes on.** G by opponent mons seen 1–2 … 6: +0.087 / +0.093 / +0.076 / +0.061 / +0.025 (TREND-DOWN,
  replicated); the axis that carries it once the others are in is the TURN; the truth changes the decision 78 % → 42 % of
  the time (k ≤ 3 vs ≥ 5) at about the same stakes. MEASURED (the mechanism rule itself is NOT DETECTED: the stakes
  difference is not bounded inside ±0.04).
- **No stage's gate is open** (R-STAGE-SEARCH, replicated rule): the FRONTIER is EMPTY; k = 6 is GATE-WEAK (pooled
  +0.024 [+0.006, +0.044] over the critic search, per checkpoint n < 150 and bounds straddle 0). MEASURED.
- **How far: the honest belief does not reach the first step.** Give the true world weight p inside the search's average:
  the curve's first step (p 0 → any p > 0) is TIE-BREAKING among the 60 % of states whose 8-world honest leaf ties (POST
  HOC FINDING, §5.1); beyond it the search needs ≈ **0.2–0.35** of weight on the true world. The network's own belief worlds
  (FN) buy nothing (+0.006 ≈ F) and sit below the step. **RED TEAM (post hoc): the "0.2–0.35" is the 8-rollout leaf's
  quantisation, NOT a belief threshold.**
  - The truth can move an untied honest pick at p ≤ 0.1 on 3 of 1,477 states. The moving weights are 1/2, 1/3, 1/4, 1/5:
    ratios of eighths.
  - On the continuous 4-turn leaf the curve rises smoothly from p = 0.05 (+0.009) and has no step.
  - The network's SPECIES belief is 2.3× Smogon's (0.41 vs 0.18 of never-seen slots hold a true species). Its value is
    +0.003 at the 8-rollout leaf and +0.008 [+0.000, +0.017] at the 4-turn leaf (a boundary). The Bayes posterior of OUR closed 719-team pool recovers 0.80 of G
  (it identifies the team), which no ladder opponent offers.
- **Maturity to the plateau does not bring honest search closer** (the maturity battery, `design_hidden_info_binding.md` §12.13.7; the plateau step READ 2026-10-25, README §7; SELF-REFEREE). The record (the plateau bank, [25M, P\*], 719 states): G_F +0.061 → +0.058, GAP-SHRINKS NOT DETECTED (−0.003 [−0.020, +0.015]); BELIEF-RISES NOT DETECTED; FRONTIER-STATIC. The interim replication reads GAP-SHRINKS (−0.030 [−0.047, −0.012]), not BANK-DEPENDENT by the registered rule. Honest search stays ≤ the policy at P\*. The first LEAN trajectory (read 2026-10-19), on 287 frozen states at 25M / 100M / 175M / 225M:
  - G_F (oracle − honest, LEAF-K8) +0.075 / +0.062 / +0.055 / +0.053; the change −0.022 [−0.049, +0.004] is NOT DETECTED.
  - Honest search's own gain over the policy stays ≤ 0 (IMP_F −0.012 / −0.022 / −0.004 / −0.011).
  - The point decline is the oracle's edge over the policy falling (ΔIMP_O −0.021 [−0.046, +0.006]), not honest search rising (ΔIMP_F +0.001); at the 4-turn leaf the gap is flat (post hoc).
  - The network's belief is worth p ≈ 0.01 on the true world at every anchor, against p\* ≈ 0.2–0.3 (BELIEF-RISES NOT DETECTED).
  - Every stage's gate is CLOSED at every anchor; the frontier is NOT READ in LEAN.
  - What would enable search is the same at 225M as at 25M: a world sampler that puts ≈ 0.2–0.3 of weight on the true world.
- **With playouts, only identifying the opponent's whole team pays** (`design_world_sampler.md` §6.2, stage 1, READ 2026-10-21; registered; PROVISIONAL, pool opponents, 225M). POOL +0.038 [+0.022, +0.055] over F, r 0.72; four open-population samplers NOT DETECTED (≈ ±0.015); the network's species × real sets, far better calibrated, gains nothing. So partly right worlds do not carry the value at the 8-rollout leaf: what search needs is exactness, which no open source in hand delivers.
- **The world sampler is the binding piece, and within it the SETS** (`design_world_sampler.md` §6.1; stage 0, zero playouts, post hoc, DESCRIPTIVE; training-pool opponents, so optimistic for the ladder).
  - The exact never-seen species set:

    | source | mass on the exact set |
    |---|---|
    | F (today's Smogon draws) | 0.020 |
    | joint Smogon teammates | 0.045 |
    | the ladder-replay corpus | 0.125 |
    | the network at X26 225M | **0.31** (25M: 0.21) |

    The network's species belief matured while its value did not.
  - Every seen mon's hidden moves right: F 0.011, the network 0.024, at most 0.035 for any candidate. So a world exact on both is drawn ≤ 0.25 % of the time, about 100× short of 0.2.
  - Next: stage 1 (`design_world_sampler.md` §6.2, ≈ 12–24 CPU-h) asks whether partly right worlds carry the value instead, arm by arm: joint Smogon teammates, a team-level corpus model, and the pool ceiling.
- **Information is a SEARCH lever, not a POLICY lever, at this maturity** (the knock-in forks, `design_hidden_info_binding.md`
  §8.7; PROVISIONAL: +3M steps from one 15M parent, no placebo). A network TRAINED 3M more steps with the opponent's true
  moves, items or never-seen species in its observation is no stronger than its control: moves −0.38 pp [−1.86, +1.10], items
  +0.13 [−1.61, +1.86], species −0.59 [−1.77, +0.58] (NULL); spreads +0.34 [−2.51, +3.18] (NOT DETECTED). Withheld at play,
  the moves and species forks are ≈ 2.4–2.6 pp WORSE than the control (descriptive): they learned to rely on the told facts,
  not to gain from them. So belief work pays, if anywhere, as a world sampler for search (G +0.070), and privileged
  information, if ever used in training, belongs in the critic only.
- **An ANCHORED acting rule removes honest search's harm but buys no gain** (L1, `../research_state/measurements/search_l1_l5_2026-10-20/`;
  pre-registered; fitted on Tier 1's 1,477 states, read held out on E2's 1,108 and X26's 287 × 4 anchors; MEASURED,
  PROVISIONAL).
  - **The rules:** piKL (argmax Q + λ·log π), Gumbel's prior-aware scale (logit + 58·c·q̃), and a deviation threshold
    (search's pick only if its margin exceeds δ). Fitted: λ 0.1 / 0.015, c 0.1 / 0.2, δ 5/32 / 0.02 (LEAF-K8 / TRUNC-4-K8).
  - **X26 (held out across policies):** the plain argmax −0.012 [−0.020, −0.004] (LEAF-K8) becomes −0.003 / −0.004 /
    −0.007; the anchor's gain over the argmax +0.006 to +0.010, clear on 4 of 6 cells. No grid value beats the policy,
    even chosen on X26 itself (best +0.004 [−0.001, +0.009]).
  - **E2:** the 4-turn leaf is already +0.010 under the argmax; anchored +0.011 to +0.012, the anchor's own gain +0.000
    to +0.002 (not clear). Registered verdict PAYS-SCOPED (one cell: piKL, TRUNC-4-K8, lower +0.001 at 98.33 %).
  - **The oracle world wants no anchor** (fitted λ 0.02 / 0.001): N5 holds there, and the honest-fitted anchor would
    forfeit ≈ 0.02 of the oracle's +0.073. Sampling π′ instead of taking its mode is no better.
  - **Reading:** Ataraxos's no-anchor row (search below its policy) replicates; its anchored row (well above) does not,
    because the honest leaf holds almost nothing the policy lacks. The acting rule does not bind; the sampler does.
  - **Replication (plateau bank, 2026-10-25, SELF-REFEREE):** the PAYS-SCOPED cell replicates (piKL, TRUNC-4-K8 +0.0097
    [+0.0025, +0.0169], 98.33 % lower +0.0010; the anchor's own part +0.0025 [−0.0039, +0.0089]), all of it at the 25M
    anchor (+0.0193) and none at P\* (+0.0001): the 4-turn leaf's weak positive belongs to the less mature policy.
- **Gen 3 OU sits near Skat on Long et al.'s PIMC map** (L5, same folder; DESCRIPTIVE, proxies). Leaf correlation 0.97
  at the 35 bank states whose playouts all end in their own turn (Skat / Hearts 0.8 to ≈ 1.0), falling to 0.02 eight
  or more turns from the end. Disambiguation (P(a new true opponent species or move per own turn)) 0.41 overall: 0.70 in
  turns 1–7, 0.19 after turn 22 (Skat ≈ 0.6). The bias proxy (0.06) is "the searched side is about to lose". In Long et
  al.'s synthetic map this region is where PIMC loses least, so strategy fusion and non-locality do not explain the
  honest null.
- **The opponent's own past choices identify the true world** (L3, `../research_state/measurements/search_l3_2026-10-20/`;
  pre-registered; 283 stage-1 states at X26 225M; forward passes only; MEASURED, PROVISIONAL, one model and bank).
  - **The likelihood:** Π over the opponent's observed past choices of π_X26(choice | the opponent's own row in world
    w). The OPPONENT's seat: its world team, our team as it saw it. Unobserved (slot-rewritten) and root-round
    choices are excluded.
  - **The truth vs today's honest worlds (`dice="rolls"`, legality on, J = 8):** AUC 0.692 [0.655, 0.728], WORKS.
    Snapshot opponents 0.796, bots 0.588.
  - **Not familiarity:** a confidence-only control reads 0.481.
  - **The truth's weight (COUNTERFACTUAL):** if the truth were inserted among the 8 honest worlds, the likelihood would
    give it 0.249 (0.360 against snapshots). This is not the dose curve's p (§5.1), the true world's weight inside the
    worlds search actually plays. No honest set holds the truth (exact-world mass ≤ 0.0025 per draw), and p_truth falls
    as proposals are added. What search needs is the near-world mass after a resample (stage 1c), not this figure
    (the L3 review, F3).
  - **Caveats (the L3 review, `search_l3_2026-10-20/REVIEW.md`):** the pooled WORKS is carried by the snapshot stratum
    (greedy earlier X26 snapshots: the model nearly is the opponent). The bot stratum alone is WEAK (0.588 [0.536,
    0.641]), and by L3's own rule it would not have built 1c (lift_moves −0.005). The read is **PROVISIONAL-POOL**: one
    pre-plateau anchor (225M) and pool opponents only, so pool familiarity is not ruled out by the confidence control.
  - **The familiarity counter-test** (`search_l3_2026-10-20/counter_test/`, pre-registered `b39c9cdc`; NOT-DETECTED):
    on the 82 of 283 states where another pool team fits every reveal, a pool DECOY beats honest worlds clearly (AUC
    0.575 [0.517, 0.631]). The truth beats its decoys only inconclusively (0.578 [0.489, 0.664]), so about half of L3's
    separation there is pool-ness (share 0.52 [0.15, 0.95]; real-team coherence is not separated from familiarity).
    Snapshots: the truth is above its decoys (0.639 [0.518, 0.754]). Bots: decoys match the truth (0.589 vs 0.588).
    The off-pool replication (the review's A4) is required before L3 or stage 1c steers anything.
  - **The off-pool replication at P\*** (`search_offpool_bank_2026-10-20/`, pre-registered `7642df0f6`): 318 states of
    P\* against its plateau roster, every opponent on a held-out ladder team.
    - Truth vs honest: snapshots 0.794 [0.753, 0.834] (unchanged from L3), bots 0.472 [0.410, 0.531] (NO-EVIDENCE),
      pooled 0.633 WORKS: **REPLICATED-SELF-PLAY-ONLY**.
    - Off-pool ladder decoys: 0.503 [0.462, 0.543] against honest worlds (**NO-LIFT**), and the truth beats them
      (0.611 [0.543, 0.680]; snapshots 0.729, bots 0.504).
    - **Reading:** the pool-bank bot signal and the pool decoy's lift were pool memory. What survives off-pool is
      truth-specific, and only where the opponent model nearly is the opponent. For a non-self opponent our network
      as the opponent model shows nothing. A behaviour-cloned HUMAN model as the judge (R-b2, 2026-10-24) shows
      nothing either: bots 0.459 [0.406, 0.510], ΔAUC vs P\* −0.013 [−0.068, +0.045] (HOM-NONE).
    - The bot stratum holds only 6 P\* losses (FINDING 1 there).
  - **The limit:** re-weighting the 8 honest worlds moves the share of correct unrevealed moves by only +0.005
    [−0.004, +0.013] (+0.014 [+0.005, +0.024] against snapshots); exact move sets and never-seen species are not
    detected.
  - **Reading:** inference from the opponent's play works as the literature says. What binds is that no honest
    proposal is near the truth: stage 1c must propose more or better worlds (64 F proposals, or the team-level corpus
    (c)) and let this likelihood choose. Against a ladder human the right opponent model is a behaviour clone
    (UNVERIFIED how our network fares there).
- **More or team-level proposals, weighted by the opponent's choices, are only slightly nearer the truth, and honest
  search over them does not pay** (stage 1c, `../research_state/measurements/search_1c_2026-10-20/`; pre-registered,
  AMENDMENT A; 285 stage-1 states at X26 225M; MEASURED, PROVISIONAL-POOL).
  - **Nearness (phase A, zero playouts), against today's 8 F worlds** (ΔH = the change in the share of the truth's
    binding hidden facts held): 128 F worlds +0.007, 128 corpus worlds (STRICT) +0.019, both +0.013.
    - All NEARER, none MATERIAL (≥ 0.05).
    - All SELF-PLAY-ONLY: carried by the snapshot stratum, the bot stratum NOT-DETECTED.
  - **What carries what:**
    - the corpus brings the never-seen species by itself (+0.095 of them; the exact species set 0.017 → 0.065);
    - the likelihood's own pick is +0.008 among F worlds (the every-move-exact share 0.011 → 0.016) and nothing among
      corpus worlds;
    - against bots it pulls the corpus worlds away (−0.008);
    - 128 proposals are no better than 64.
  - **The exact-world mass stays ≈ 0** (≤ 0.0003) against the dose curve's ≈ 0.2.
  - **The playouts (phase B, L1's piKL anchor):** every arm NOT-DETECTED against the policy (C128L +0.007 [−0.003,
    +0.017] at LEAF-K8, F128L −0.001, M256L −0.002; n 285). Holm rejects none, and none beats today's 8 F worlds. The
    plain argmax again plays below the policy (F8 −0.020); the anchor removes that harm.
- **What does not help:** more rollouts as critic-training labels, a per-action head on noisy oracle labels (−0.004), the
  proof-first 2·SE acting rule, trusting the policy prior at Gumbel's default scale, shared-seed dice pairing beyond one
  turn, a privileged critic, telling the frozen network the true moves (+0.001), training it 3M steps with them (−0.38 pp), the network's belief as a world sampler,
  the honest sharpening (prior-mode moves), the pool's per-species sets, Foul Play's hand evaluator as the leaf in
  honest worlds (no better than the critic), and an acting rule anchored to the policy (it removes the argmax's harm
  in honest worlds but does not beat the policy held out, L1), and weighting 128–256 honest or real-team proposals by the opponent's own choices
  (slightly nearer worlds, no playout gain; stage 1c).
- **What the literature says search needs** (`../research_state/literature/hidden_info_search_review_2026-10-20.md`;
  a dated review, every claim VERIFIED in its primary source or marked; the inferences are the reviewer's judgement,
  not measurements). One-ply determinized search is the regime where the classic failures of perfect-information Monte
  Carlo are smallest: there is no strategy fusion when one action is shared by every world. The sound methods
  (DeepStack, ReBeL, Student of Games, Obscuro) need a belief over, or an enumeration of, every private state, and are
  outside our regime by many orders of magnitude. The closest in-regime precedent is Ataraxos (Stratego, 2025): sampled
  worlds × every legal move, value-net leaves, a belief network trained on the self-play truth, and a KL ANCHOR to the
  policy. Without that anchor its search played 362 Elo BELOW its own policy, our symptom. Card-game programs gain
  most by weighting worlds by the opponent's own past choices (Skat, bridge), the one honest source of evidence about
  unrevealed MOVES. Every published method and every winning Pokémon search bot samples from a belief that is right IN
  DISTRIBUTION (the format's generator, a set list, team sheets, a network trained on the truth). None tests a belief
  worth ~1 % of the truth. In Skat, all weight on the truth was not the best play. So the review reads the "p ≈ 0.2–0.3"
  of §5.1 as what a MISCALIBRATED prior costs, not as a requirement on a calibrated one (its FINDING F1; untested).
  Our "the prior hurts" (N5) was measured in the ORACLE world only (F2).
- **Open, ranked by decision value per cost** (the review's §10):
  1. **L1 READ 2026-10-20: PAYS-SCOPED, and in substance the anchor removes harm, not adds gain** (the bullet above).
     **Replicated on the maturity battery's plateau bank 2026-10-25** (the cell REPLICATES, +0.0097, lower +0.0010,
     SELF-REFEREE; 25M anchor +0.0193, P\* +0.0001; the anchor's own part +0.0025, not clear). The module's honest
     default stays; an anchor is worth adopting only as a guard against the argmax's harm, never as the source of a gain.
  2. **L2, a descriptive amendment to the RUNNING stage 1, before its read:** does an arm's IMP follow the
     log-likelihood its worlds give the truth, rather than its exact-world mass? Zero cost; time-critical. (C_pool is
     both the only exact and the only calibrated arm for a pool-opponent bank, so "only C_pool pays" cannot separate
     the two readings.)
  3. **L3 READ 2026-10-20: WORKS; off-pool at P\* SELF-PLAY-ONLY** (pool bank AUC 0.692; off-pool snapshots 0.794,
     bots 0.472 NO-EVIDENCE; the bullet above). Its registered
     consequence: build stage 1c (`design_world_sampler.md` §4 (f), §6.2), with a proposal set large or good enough
     to hold near-true worlds, since re-weighting today's 8 moves almost nothing. **Stage 1c's PHASE A READ
     2026-10-20, PHASE B 2026-10-21** (`design_world_sampler.md` §6.3): 128 F + 128 C_ladder proposals weighted by dice ×
     the likelihood are NEARER the truth (ΔH +0.007 / +0.019 / +0.013), SELF-PLAY-ONLY, the exact-world mass ≈ 0; the
     playouts read NOT-DETECTED on every arm (Holm: none pays). NEARER BUT NOT PAYING.
  4. **L4, a calibrated-by-construction bank:** opponents whose teams are drawn by F's own sampler, then O and F. Does a
     right but diffuse prior suffice? ≈ 1–2 agent-days + 10–20 CPU-h.
  5. **L5 READ 2026-10-20 (descriptive):** leaf correlation 0.97 near the end, disambiguation 0.41 (0.70 early): near
     Skat on Long et al.'s map (the bullet above).
  6. **L6, the Bayesian one-turn matrix** (Q11), after L1 or a sampler pays.

  Still open from the study's red-team list: the second referee (≈ 3–5 CPU-h), species-only never-seen worlds (F+Usp,
  ≈ 15 CPU-h), Tier 2, and the maturity battery at plateau (RUNNING, §7 Q4). E2 is READ (§16.12: U-FIRST,
  scorer-dependent, PROVISIONAL; no more CPU on U vs M at 15M).

---

## 2. WHAT WORKS

| # | claim | tag | number [95 %] | pointer |
|---|---|---|---|---|
| W1 | **An oracle-world rollout leaf improves decisions; the critic barely does.** LEAF-K8 over the policy's pick (end-state / production); critic one-ply; K16 | PROVISIONAL (oracle world) | LEAF-K8 **+0.081** [+0.066, +0.095] / **+0.071** [+0.057, +0.087]; critic +0.026 [+0.014, +0.039] / +0.012 [+0.000, +0.023]; K16 +0.098 / +0.082. R-LEAF (K8 − critic) +0.055 [+0.040, +0.068] / +0.059 [+0.045, +0.075] LEAF-AHEAD | `design_value_discrimination.md` §7.11; ledger 2026-10-11 L24341; `value_disc_pilot_2026-10-11/` README §6 |
| W2 | **The first few rollouts carry most of it; the curve still climbs.** K1 / K4 / K8 / K16 | PROVISIONAL | end-state +0.035 / +0.069 / +0.081 / +0.098; R-K (K8 − K4) +0.012 [+0.004, +0.020] STILL CLIMBING. Banked rb-blob (Rustboro-era blob arm, off-distribution Lane S states): K1 +0.025 [+0.012, +0.037], K8 +0.062 [+0.045, +0.079], critic +0.008 [−0.005, +0.021] | same; ledger 2026-10-11 L24263 |
| W3 | **A 4-turn leaf then the critic keeps about half the value at a seventh of the cost.** | PROVISIONAL | TRUNC-4-K8 +0.051 / +0.037 at 2.6 / 1.7 CPU-s vs LEAF-K8 +0.081 / +0.071 at 19.6 / 11.5; R-TRUNC says NONE keeps it within 1 point. 1–2 turns buy nothing over the critic | pilot README §6 |
| W4 | **The acting rule: uniform allocation, ARGMAX over ALL legal actions.** The value sits as much in the policy's confident states (q5 +0.075 / +0.072) as its torn ones (q1 +0.078 / +0.090) and in moves the policy ranks low | PROVISIONAL | the module's default (`SearchConfig`: uniform, depth 4) | `design_value_discrimination.md` §7.11 |
| W5 | **An honest one-ply CRITIC search is the one fair scorer clear of 0.** | PROVISIONAL, one seed, builder-sensitive | end-state +0.014 [+0.002, +0.026]; 4-turn K8 +0.021 [+0.008, +0.035]; production +0.005 / +0.007 (not clear). NOT shown in games (S0b not run) | `design_value_discrimination.md` §7.14, §13.0; ledger 2026-10-12 L24578 |
| W6 | **Late, honest search beats the policy (and, pooled, the critic search).** | PROVISIONAL, DESCRIPTIVE | all six seen: TRUNC-4 over the critic search **+0.026** [+0.004, +0.051], over the policy **+0.041** [+0.018, +0.066]; per checkpoint end-state +0.022 [−0.005, +0.052], production +0.031 [−0.001, +0.075]. ≤ 7 mons alive: leaf over the policy +0.022 [+0.007, +0.038] but not over the critic search | `hidden_info_by_reveal_count_2026-10-12/` README §1, `tables.md`; ledger 2026-10-12 L24663 |
| W7 | **A zero-cost scorer exists:** the network's own ride-along A head, argmax, no rollout | PROVISIONAL, DESCRIPTIVE, end-state only | +0.010 [+0.001, +0.020]; one registered replication owed before anything is built on it | pilot README §12; ledger 2026-10-11 L24485 |
| W8 | **The lever is real if the worlds knew the truth:** worlds with the seen mons' TRUE moves (CT′) | PROVISIONAL | LEAF-K8 +0.042 [+0.028, +0.058] / +0.018 [+0.004, +0.033]; beats its own critic on end-state by +0.025 [+0.009, +0.040] (DESCRIPTIVE); closes ½ / ⅕ of the oracle gap | pilot README §14 |
| W9 | **Partial truth in the belief is worth a lot, once it is more than a tie-breaker.** Weight p on the true world among the honest ones | MEASURED (Tier 1, M8 on F; CONCAVE, CONTENT-MATTERS, CRITIC-INFO-LIMITED), with a POST-HOC qualifier | on F: +0.002 (p = 0) → +0.021 (0.1) → +0.032 (0.2) → +0.053 (0.35) → +0.062 (0.5) → +0.072 (1.0); the true world beats a random honest world at the same weight (M5 − M5W +0.028 [+0.024, +0.032] at 0.35). POST HOC: the whole step to p = 0.1 is reached at p = 0.001 (TIE-BREAKING, §5.1), so the dose proper runs from ≈ 0.1 to 0.75. The step and the curve REPLICATE on the blind replica's own F worlds (ties 0.60, a jump of +0.024 at p = 0.001; every point's difference from the original's contains 0; post hoc, §16.9). **RED TEAM (§16.10): the "≈ 0.1–0.75 / 0.2–0.35" reading is the 8-rollout leaf's QUANTISATION** (the moving weights are 1/2, 1/3, 1/4, 1/5; at p ≤ 0.1 the truth can move an untied pick on 3 of 1,477 states). On the continuous 4-turn leaf the curve has no step: +0.002 (0) → +0.009 (0.05) → +0.014 (0.1) → +0.023 (0.2) → +0.038 (1) | `design_hidden_info_binding.md` §16.8, §16.9, §16.10; `hidden_info_tier1_2026-10-13/` README §7, `red_team.md`; the pilot's curve: `hidden_info_by_reveal_count_2026-10-12/` README §4 |
| W10 | **An existence proof that determinised search wins here (PRE-ERA, a width meter).** Foul Play (decoupled UCT over a hand-written evaluator, Smogon-determinised sets) vs the 75M policy | MEASURED, PRE-ERA | our win rate **0.388** [0.288, 0.497], n = 80, ~1.4M visits per decision; width and opponent-set knowledge NOT SEPARABLE (width slope +0.027 per ln-visit, CI covers 0; knowledge delta +0.000 [−0.15, +0.15]). Later (PRE-ERA Kakuna, greedy vs Foul Play at 1000 ms): 0.490 [0.422, 0.559], NOT DETECTED | `UNDERSTANDING.md` §3.1; ledger 2026-09-14 L18248, 2026-09-17 L18964, 2026-09-29 L21447 |
| W11 | **A rollout leaf out-ranks the learned critic head (PRE-ERA).** K = 1/2/4/8/16: 0.543 / 0.580 / 0.613 / 0.652 / 0.671 vs 0.574 | MEASURED, PRE-ERA | DETECTED at K ≥ 8 (+0.078 [+0.027, +0.128]); the single-rollout label itself mis-orders the top1\|top2 pair 39.7 % of the time (so any pairwise level is compressed toward 0.5) | ledger 2026-09-19 L19881, L19663 |
| W12 | **Identifying the opponent's TEAM recovers most of the gap, on our closed pool.** Worlds drawn from the training-pool teams consistent with the observation (POOL), gated | MEASURED (Tier 1, R-RECOVER BELIEF-RECOVERABLE on this distribution) | r_POOL **0.80** [0.73, 0.87]: POOL IMP +0.063 vs oracle +0.076; the pool's per-species SETS alone (POOLSETS) recover −0.02: what pays is knowing WHICH TEAM, not the set priors. **At X26 225M (world sampler stage 1, registered): Δ over F +0.038 [+0.022, +0.055], r 0.72 [0.53, 0.92], on bots and own snapshots alike (a SELF-REFEREE read; on bots, E1-0's real-bot referee gives −0.006 [−0.034, +0.019] vs +0.027, UNDECIDED); its 4-turn leaf does not clear its own critic search (+0.016 [−0.001, +0.034]).** Not available on the ladder (an open population) | `design_hidden_info_binding.md` §16.8; `design_world_sampler.md` §6.2 |

---

## 3. WHAT DOESN'T WORK, and why

| # | what | tag | evidence | why (a mechanism, and whether it is measured) |
|---|---|---|---|---|
| N1 | **The old proof-first ACTING RULE** (switch only when the paired difference is resolved at 2·SE) | MEASURED (PRE-ERA, reproduced on rb-blob and in the pilot) | resolved 4.5 % / 12.5 % of pairs at R = 4 / 16; changed 1 of 250 live decisions; rb-blob 6.7 / 15.3 / 17.7 / 26.9 % at R = 4 / 8 / 16 / 32. GATE-2SE at K8 keeps +0.050 / +0.041 of ARGMAX's +0.081 / +0.071. Opening it to 0.5·SE bought 6.2× more offline, but the live read (0.4719 [0.402, 0.542]) is VOID (a recon-tag collision wrote fabricated choices into the live record) | the question "is this pair resolved" is the wrong one: decisions need the best of ~10 siblings, not a certified pair. MEASURED (ledger 2026-09-19 L19885, 2026-09-20 L19994, 2026-10-11 L24263) |
| N2 | **More rollouts as critic-training labels** (a frozen readout refit; richer-label arms; λ-returns; contested-state forks into the buffer) | MEASURED (CRN refit on rb-blob; the rest PRE-ERA) | REFIT-K8 − BASE **−0.014** [−0.032, +0.002]; K8 vs K1 label no difference; arm 10 R-rollout targets NULL at 10M; λ-return 0.9 not confirmed; fork arm Δ −0.0003 [−0.0148, +0.0136]; twelve heads on 0.567–0.578 | the readout is not label-limited (a head fitted in 1.1 s matches one trained 75M steps); the label buys calibration (ECE 0.065 → 0.032), not resolution. Whether a TRAINED trunk on sibling-level targets differs is UNVERIFIED (ledger 2026-10-06 L22512, 2026-09-19 L19663) |
| N3 | **A per-action head on noisy oracle labels** (frozen end-state body, 2 oracle playouts per action, 9,629 states) | PROVISIONAL, end-state only | R-QHEAD **−0.004** [−0.016, +0.009] NOT DETECTED; R-QHEAD-CRITIC **−0.018** [−0.032, −0.004] BEHIND (v2 builder, confirmed); validation R² +0.002 | **83 %** of the labels' sibling spread is dice noise (R² ceiling 0.17) and most of the rest is hidden information; a head that sees only our observations can learn ~0.5 % of the variance. How many labels would suffice is UNVERIFIED; a trained body is untested (ledger 2026-10-11 L24485) |
| N4 | **Shared-seed (CRN) pairing across a full playout** | MEASURED | VRF **1.057** [1.045, 1.072] at the terminal (move-move pairs 1.28, move-switch 1.04); ρ after one turn 0.44–0.53, at the terminal 0.06; the derangement VRF 1.000 [0.996, 1.005] so the unpaired variance is estimable from shared-seed data | ONE PRNG per battle: shared seeds share the STREAM, not the roll-to-event MAPPING, and the first differing draw misaligns every later one. R-PAIR reads ALIGNMENT-LOSS ⇒ the registered rule says build S1 (per-turn reseed); NOT built. The AIVAT-style branch-turn control variate buys nothing (CV-K8 − LEAF-K8 ≈ 0) (`design_value_discrimination.md` §5) |
| N5 | **Gumbel's prior-aware selection at the `mctx` defaults (c_visit 50, c_scale 0.1), and top-k restrictions / sequential halving** | PROVISIONAL | R-PRIOR **−0.013** [−0.021, −0.005] / **−0.022** [−0.031, −0.013] (K4); at K8 −0.022 / −0.031; SH-4 is NOT DETECTED at a small budget and WORSE (−0.011…−0.027) at larger ones. A post-hoc read (`target_scale_read.json`): at c_scale ≥ 1 sampling π′ recovers the full ARGMAX gain | where the leaf disagrees with the policy, the leaf is right; at 0.1 our win-scale Q (a spread of ~0.1–0.2) is too small against the policy's logits, so the rule under-trusts it. The module now defaults to c_scale 1.0 (the calibration reading is POST HOC). **ORACLE world only:** on the fully honest base the prior's effect is unmeasured, and anchored search is the literature's norm (the 2026-10-20 review, F2; its L1). **Honest world, L1 (2026-10-20, pre-registered, held out):** a prior-aware rule fitted on Tier 1 (Gumbel c 0.1 / 0.2, piKL λ 0.1 / 0.015) lifts the argmax's harm on X26 (−0.012 → −0.004 / −0.003, gain over the argmax +0.008 / +0.010, clear) but does not beat the policy on either held-out bank; in the oracle world the fitted anchor is ≈ none (λ 0.02 / 0.001), so N5 stands there (`../research_state/measurements/search_l1_l5_2026-10-20/`) |
| N6 | **Depth-1 search over the critic of the time, in fair worlds (PRE-ERA)** | MEASURED, PRE-ERA | depth-1 search COST 17–22 points of mirror win rate (honest@1 s 0.294 [0.24, 0.35]); flat across a 32× dice sweep (0.125 … 0.205): BIAS, not variance; 13× more certified overrules moved the win rate onto the null (0.5003 [0.4803, 0.5203]); no win-prob head paid as a leaf at 3 s (0.506 / 0.503 / 0.495) | search AMPLIFIES the critic's systematic error; the leaf, not the allocator, was the limit. The current-era re-measure is §2 W5–W6 (ledger 2026-08-24 L04104, L04318, 2026-08-29 L05912, 2026-09-11) |
| N7 | **The asymmetric (privileged) critic** | MEASURED, PRE-ERA, one run, no replicate | `truevalue` (V given the opponent's TRUE team) at 10M: NOT DETECTED on discrimination (turn-1–3 spread ratio 0.00 vs 0.09–0.12); +0.040 (all) / +0.088 (late) more OPTIMISTIC than its own continuation; ECE +0.04 | hypothesis only: a critic conditioned on information the policy never has is confidently wrong about what the policy will do. The information family is CLOSED for the era; the omniscient twin (X11) stays offline, never a training target (ledger 2026-09-10 · READ · *arm 5 … (the PRIVILEGED critic) on the OFFLINE … frames*; `design_q_head.md` §7) |
| N8 | **Telling the NETWORK the true moves** (frozen; and trained) | PROVISIONAL (frozen; the knock-in forks); MEASURED (X5 oracle arms, below the floor) | injected at 95 %: it flips 13 % of picks and gains **+0.001** [−0.003, +0.005] (R-NET, CRITIC-USES NOT DETECTED, 3 networks); a wrong-moves placebo costs the critic search ≈ 0.8 points. X5 oracle arms: told species +2.44 pp [+0.31, +4.57], full set +2.18 pp [+0.53, +3.84], the set over species −0.26 [−1.92, +1.40], all below the 4.57 pp replicate floor; an untrained network handed a revealed team FELL −8.1 pp. **Knock-in forks** (3M more steps of the end state with the truth at 95 % in its observation, 3 seeds per arm vs a no-change control, `design_hidden_info_binding.md` §8.7): moves **−0.38 pp** [−1.86, +1.10], items +0.13 [−1.61, +1.86], never-seen species −0.59 [−1.77, +0.58] (NULL), spreads +0.34 [−2.51, +3.18] (NOT DETECTED); MOVES-FIRST false; no placebo. Withheld at play: moves −2.35, species −2.58 pp vs the control (told − withheld ≈ +1.9, descriptive) | a USABILITY limit, not an information one. The frozen network does not convert the move content. A trained one learns to READ it (removing it costs ≈ 1.9 pp) but not to GAIN from it: it substitutes the told facts for its own inference (mechanism UNVERIFIED: distribution shift, learned reliance or capacity; §8.7 lists the separating experiments) |
| N9 | **Smogon-usage or network-belief worlds as a sufficient sampler** | PROVISIONAL | R-HONEST NOT DETECTED (+0.011 / +0.007); R-BELIEF (network species belief − Smogon) −0.010 [−0.024, +0.005] / −0.002; the network's MOVE belief closes ~⅕ (end-state) / ~1⁄20 (production) of the oracle gap (R-MOVES +0.017 [+0.001, +0.031] HELPS end-state only, NOT DETECTED on production, builder-sensitive) | the belief is barely sharper than Smogon usage (51 % / 48 % of a world's hidden moves right vs 45 %); a better species hit rate (0.45 vs 0.21) buys NO value (B′ ≈ 0) because value needs the whole world to hang together |
| N10 | **"Focus the search on torn states"** | PROVISIONAL | LEAF-K8 flat across the policy's gap quintiles (q1 +0.078 / +0.090, q5 +0.075 / +0.072); banked: torn 40 % +0.049 vs most decided 20 % +0.082. No cheap policy-confidence feature separates the decisions search flips (PRE-ERA: flip rate 0.694 [0.681, 0.707]) | the stakes are larger where the policy is confident |
| N11 | **Narrowing the state distribution to buy discrimination** (an exploiter's head) | MEASURED, PRE-ERA | sibling ranking 0.575 on its own slice and the generalist's alike | the 0.575 ceiling was four heads deep: coverage, width, loss form and data each varied |
| N12 | **Hand evaluators as the leaf (PRE-ERA)** | MEASURED, PRE-ERA | nine tried; none beat the head; the PBRS potential matches a 75M critic pooled | Foul Play's edge is evaluator, search shape or knowledge: not separated (§7) |
| N13 | **Foul Play's hand evaluator as the leaf, current era, honest worlds** (R-EVAL) | MEASURED (Tier 1) | in F: HE − critic +0.007 [−0.001, +0.016] (4-turn leaf), −0.002 [−0.011, +0.006] (one ply); in the pilot's P: +0.003 / −0.005: **not EVALUATOR-BOUND**; the hand eval also collapses oracle → honest (g_HE / g_CRIT 0.75 [0.52, 1.06]): INFORMATION-BOUND support. **RED TEAM (§16.10): WEAKENED**: the 4-turn contrast is a near miss TOWARD evaluator-bound (one-sided tail 5.5 % at the +0.0005 bar; positive on both checkpoints); not replicated | information binds; a hand leaf ≈ 0.7 points better than our critic in honest worlds is NOT excluded |
| N14 | **The seen mons' hidden MOVES as THE binding constraint** (R-MOVES) | MEASURED (Tier 1): **REFUTED (R3)** | s_M 0.13 [−0.02, 0.25], n_M 0.42 [0.31, 0.54] vs s_U 0.31, n_U 0.82 (U dominates on both sides, paired: +0.19 [+0.03, +0.35], +0.41 [+0.26, +0.55]); the moves' dose helps a little (D90 − F +0.011), a wrong set hurts (W90 − F −0.013). **On E2's fresh competent-opponent states too** (§16.12): R3 U-FIRST (scorer-dependent), s_M 0.16 (B3 fails), n_M 0.58 (B4 holds: necessary, not sufficient) | the never-seen Pokémon carry more; a move-inference programme is not the lever (`design_hidden_info_binding.md` §16.8) |
| N15 | **The network's own belief as the world sampler on the honest base** (FN) | MEASURED (Tier 1) | FN LEAF-K8 +0.006 = F's +0.006 (4-turn +0.013 vs +0.005); its move hit 0.513 vs Smogon 0.508, species (a never-seen slot holding a TRUE never-seen species) **0.411 vs 0.180** (the published 0.172 vs 0.058 counted a species against the truth of the slot it was DRAWN for: the 2026-10-15 builder-record GIGO, `design_information_levers.md` §7.6; the red team's count from the world teams agrees). FN − F (red team, §16.10 of the study): +0.003 [−0.006, +0.012] at LEAF-K8, +0.008 [+0.0001, +0.017] at TRUNC-4-K8 (a BOUNDARY result) | barely sharper than Smogon on MOVES; on SPECIES 2.3× sharper, and that buys at most ≈ +0.017 at any scorer (the whole world must hang together). **At X26 (`design_world_sampler.md` §6.1, stage 0, descriptive):** the exact never-seen species set gets 0.21 (25M) → 0.31 (225M) of the belief's mass, but every seen mon's hidden moves are right in only 0.024 of its worlds (F 0.011). The exact-world mass is 0.0005: the moves, not the species, keep it from the truth |

| N16 | **An HONEST-world one-ply read as a STARVATION detector / anti-starvation target** (which starved near-best moves are good?) | DESCRIPTIVE, POST HOC (on Tier 1's arm F and the pilot's truth; `design_collapse_and_counterfactuals.md` §3.5) | on decisive turns, flagged starved moves are truly near-best at precision 0.260 (F, 8 worlds) vs 0.194 chance and 0.501 for the ORACLE 8-rollout leaf; playing F's flagged move instead of the policy's top choice **−0.068** [−0.105, −0.031] (oracle +0.137 [+0.104, +0.169]); the critic one-ply on F worlds −0.035 | the value of the moves the policy starves is hidden information (the never-seen mons first, N14); at 8 samples "within 0.1 of the best" is "tied with the best", and F's per-action SE is 0.24. A counterfactual TARGET from honest one-ply values would feed the wrong moves |
| N17 | **Open-population world samplers as the honest base** (joint Smogon teammates; a ladder-corpus team model, λ = n / (n + 89); the forum corpus minus near copies of the truth; the network's species × forum per-species sets) | MEASURED (world sampler stage 1, registered; PROVISIONAL, pool opponents, 225M) | Δ over F at LEAF-K8: −0.004 [−0.019, +0.011] / +0.002 [−0.013, +0.018] / −0.008 [−0.023, +0.007] / −0.004 [−0.023, +0.014], all NOT DETECTED, none SEARCH-ENABLED; C_pool − C_forum_strict +0.046 [+0.029, +0.063] | none makes a world EXACT (q_SM ≤ 0.001 against POOL's 0.82); the network-species arm's far better calibration (true never-seen set in 44 % of worlds, +3.3 nats) buys nothing, so at this leaf partial correctness is not value (L2 EXACT-ONLY between arms). The corpora rarely hold the opponent's team (λ ≈ 0.14–0.29) |
---

## 4. THE FRONTIER MAP (by stage of the game)

**How to read it.** The registered Tier-1S read (`design_hidden_info_binding.md` §16.8, §12.13.4): G = oracle IMP − honest IMP
on the FULLY honest base F (every hidden fact drawn and gated); "honest 4-turn" = F's TRUNC-4-K8 over the policy; "vs critic" =
F's TRUNC-4-K8 over F's one-ply critic search (the R-STAGE-SEARCH contrast); the class column is R-STAGE-CLASS with the
two-point Shapley credit φ_X = ½(reveal-only gain + hide-only cost) of the never-seen mons (U) and the seen mons' moves (M);
the dose column is M8 LEAF-K8 at p = 0 → 0.35 → 1 (the true world at weight p, F's worlds at 1 − p). Pooled over both
policies, registered 600 + the k-stratified supplement (≥ 160 per k bin per policy before exclusions), battle-clustered 95 %.
MEASURED, PROVISIONAL at 15M, NOT replicated. The replica holds no supplement, no U arm and no critic scorer. Its sub-read of
the k-gradient on the registered states DIFFERS: −0.011 [−0.022, +0.0003] per SD, against the original's −0.023 [−0.033, −0.013]
on the same states (study §16.9). The axes are correlated (k–material −0.71, k–turn +0.66); jointly only
the TURN keeps a slope (R-STAGE-AXIS). The pilot's earlier descriptive version (on its leakier builder) is
`hidden_info_by_reveal_count_2026-10-12/` (§16.7 of the study).

| stage | n | **G (oracle − honest)** | honest 4-turn | 4-turn vs critic search | which class leads (φ_M, φ_U) | gate | dose p 0 → 0.35 → 1 |
|---|---|---|---|---|---|---|---|
| k = 1-2 | 331 | **+0.087 [+0.066, +0.108]** | +0.003 | +0.007 [-0.012, +0.024] | U-LED (φ_M +0.017, φ_U +0.065) | CLOSED | -0.004 → +0.055 → +0.083 |
| k = 3 | 299 | **+0.093 [+0.071, +0.116]** | -0.011 | -0.010 [-0.028, +0.010] | U-LED (φ_M +0.022, φ_U +0.056) | CLOSED | -0.026 → +0.050 → +0.067 |
| k = 4 | 286 | **+0.076 [+0.052, +0.101]** | +0.007 | +0.001 [-0.021, +0.021] | U-LED (φ_M +0.014, φ_U +0.048) | CLOSED | +0.013 → +0.077 → +0.089 |
| k = 5 | 285 | **+0.061 [+0.042, +0.079]** | -0.015 | -0.014 [-0.033, +0.006] | SIDES-DISAGREE (φ_M +0.020, φ_U +0.022) | CLOSED | -0.004 → +0.026 → +0.057 |
| k = 6 | 276 | **+0.025 [+0.010, +0.042]** | +0.029 | +0.024 [+0.006, +0.044] | — (nothing unseen) | **WEAK** (pooled 95 % only; per policy n 145 / 131 < 150) | +0.035 → +0.056 → +0.060 |
| material 12 alive | 257 | **+0.094 [+0.072, +0.115]** | -0.009 | -0.016 [-0.034, +0.004] | SIDES-DISAGREE (φ_M +0.025, φ_U +0.060) | CLOSED | -0.018 → +0.032 → +0.076 |
| material 10-11 | 468 | **+0.071 [+0.054, +0.090]** | +0.015 | +0.004 [-0.011, +0.020] | SIDES-DISAGREE (φ_M +0.019, φ_U +0.042) | CLOSED | +0.010 → +0.071 → +0.081 |
| material 8-9 | 357 | **+0.081 [+0.060, +0.103]** | -0.004 | +0.011 [-0.010, +0.031] | SIDES-DISAGREE (φ_M +0.022, φ_U +0.043) | CLOSED | +0.003 → +0.066 → +0.083 |
| material ≤ 7 | 395 | **+0.041 [+0.027, +0.058]** | +0.000 | +0.001 [-0.014, +0.017] | SIDES-DISAGREE (φ_M +0.011, φ_U +0.020) | CLOSED | +0.006 → +0.034 → +0.047 |
| turn ≤ 7 | 346 | **+0.095 [+0.073, +0.116]** | -0.001 | -0.003 [-0.020, +0.017] | U-LED (φ_M +0.022, φ_U +0.064) | CLOSED | -0.013 → +0.054 → +0.082 |
| turn 8-14 | 451 | **+0.078 [+0.060, +0.096]** | -0.003 | -0.000 [-0.015, +0.015] | U-LED (φ_M +0.017, φ_U +0.055) | CLOSED | -0.008 → +0.053 → +0.070 |
| turn 15-21 | 335 | **+0.069 [+0.050, +0.089]** | +0.001 | -0.003 [-0.023, +0.017] | NOT DETECTED (φ_M +0.030, φ_U +0.026) | CLOSED | +0.013 → +0.060 → +0.082 |
| turn ≥ 22 | 345 | **+0.033 [+0.019, +0.049]** | +0.014 | +0.012 [-0.004, +0.030] | NOT DETECTED (φ_M +0.007, φ_U +0.007) | CLOSED | +0.019 → +0.044 → +0.052 |

**What the table says.**
1. **The gap falls through the game** (R-STAGE-VOI: TREND-DOWN on k and turn, TREND-UP on material left, each replicated in
   sign on both policies); jointly the TURN carries it (partial slope −0.014 [−0.024, −0.003] per SD; k and material do not
   once it is in).
2. **Mostly a frequency effect, descriptively:** the truth changes the honest pick 78 % of the time at k ≤ 3 vs 42 % at
   k ≥ 5, at about the same stakes (0.116 vs 0.103); the registered mechanism rule reads NOT DETECTED because the stakes
   difference (+0.013 [−0.021, +0.045]) is not bounded inside ±0.04.
3. **Early, the never-seen Pokémon lead** (U-LED at k ≤ 4 and turn ≤ 14); in the middle and late the two classes cannot be
   separated (SIDES-DISAGREE / NOT DETECTED); the seen mons' moves never LEAD at any stage. The pilot's descriptive "moves lead
   at k = 4" (on its leakier builder) does not replicate on the fully honest base.
4. **The frontier is EMPTY.** No bucket's honest 4-turn search beats both the policy and the critic search on each policy
   separately; k = 6 passes pooled at 95 % only (and cannot reach GATE-OPEN as built: F's exclusions leave n < 150 per policy).
   The cheapest labels sit where there is least to gain (`design_value_discrimination.md` §13.10).

---

## 5. HOW FAR WE ARE

### 5.1 The belief-quality dose-response

Search averages K worlds; the true world gets weight p, the honest worlds share 1 − p. One set of per-world outcomes gives
the whole curve by re-weighting (no extra playout). **p is a weight on the exact true world, NOT a probability the network
holds**; "today's p" is the p-EQUIVALENT in value units. Registered read on the FULLY honest base F (`design_hidden_info_binding.md`
§16.8, §6.9), pooled, MEASURED (PROVISIONAL at 15M):

| | p = 0 | 0.001 (post hoc) | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| leaf K8 (M8) | +0.002 | +0.021 | +0.021 | +0.032 | +0.053 | +0.062 | +0.071 | +0.072 |
| one-ply critic search (M8) | +0.001 | | +0.004 | +0.008 | +0.009 | +0.010 | +0.014 | +0.015 |
| leaf, literal K = 5 (one playout per world) | +0.001 | | +0.012 | +0.020 | +0.027 | +0.028 | +0.028 | +0.025 |
| same with a WRONG (random honest) world in the true one's place | +0.001 | | −0.001 | +0.001 | −0.001 | −0.001 | −0.001 | +0.000 |
| same with the MOST wrong honest world | +0.001 | | −0.002 | +0.000 | −0.002 | −0.002 | −0.002 | −0.002 |

- **🚨 POST-HOC FINDING: the first step is TIE-BREAKING, not dose.** An 8-world leaf is a mean of 8 win / loss outcomes, so the
  honest argmax is tied in **60 %** of states; ANY positive weight on a second world breaks the ties by it. The curve reaches
  +0.021 at p = 0.001 and is flat to 0.1 (`hidden_info_tier1_2026-10-13/posthoc_tiebreak.json`). The registered reads that
  sit on that step — **P-STAR = 0.1** (all three comparators), **TODAY(FN) = 0.009** [≤ 0.032], **WITHIN-REACH**,
  DETERMINIZATION-FIRST at k = 4 / turn 15–21 — are computed as registered but do NOT mean a 10 % belief suffices. The pilot's
  curve (§16.7 of the study, +0.009 → +0.034 at 0.1) carries the same step.
- **Beyond the step** the dose rises from p ≈ 0.1–0.2: half its further rise by p ≈ 0.30, so a search would need ≈ **0.2–0.35**
  of weight on the TRUE world. The network's own belief worlds (FN, +0.006 ≈ F's +0.006) do not even reach the step.
  **RED TEAM (§16.10, post hoc): that threshold is the 8-rollout leaf's resolution, not a belief requirement.**
  - The leaf moves in eighths, so the first weights at which the truth can overturn an UNTIED honest pick are 1/5, 1/4, 1/3
    and 1/2 (only 3 of 1,477 states at p ≤ 0.1).
  - On the continuous TRUNC-4-K8 leaf the same re-weighting rises with no step:

    | p | 0 | 0.001 | 0.01 | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 | 1 |
    |---|---|---|---|---|---|---|---|---|---|
    | IMP | +0.002 | +0.003 | +0.004 | **+0.009** [+0.000, +0.018] | +0.014 | +0.023 | +0.029 | +0.034 | +0.038 |

  - It covers half its rise by p ≈ 0.15. How much belief search needs depends on the leaf's resolution.
  - **The literature review (2026-10-20, F1; judgement, untested):** the curve mixes the truth into F, a prior that is
    MISCALIBRATED for these states' opponents. It prices a wrong prior; it does not say what a calibrated but diffuse
    belief would need. Every published sampled-world search uses a belief that is right in distribution. L2 and L4 test
    this.
- **Along X26's maturity (the battery's first LEAN trajectory, 2026-10-19; PROVISIONAL, interim bank; §1).** The same curve on X26's 25M / 100M / 175M / 225M checkpoints, 286 frozen states:
  - TODAY (registered, LEAF-K8) is 0.00 / 0.01 / 0.00 / 0.01: below the tie-break step at every anchor (BELIEF-RISES NOT DETECTED, +0.011 [−0.034, +0.178]).
  - P-STAR against the policy (post hoc, R-DOSE's definition) is 0.25 / 0.30 / 0.20 / 0.30.
  - On the continuous 4-turn leaf (post hoc, not registered) TODAY drifts 0.02 → 0.18 [0, 0.36], while P-STAR drifts out 0.35 → 0.50 as the oracle's ceiling falls.
  - The distance to p\* does not close with training. At the plateau (README §7, registered): TODAY 0.001 → 0.014 [0, 0.046] on the plateau bank's 717 FN-common states, 0.012 → 0.014 on the interim's 722; BELIEF-RISES NOT DETECTED on both.
- **Registered shape reads:** CONCAVE (Δ² +0.025 [+0.020, +0.031]); CONTENT-MATTERS (the true world beats a random honest one at
  the same weight: +0.028 [+0.024, +0.032] at 0.35); OVERCONFIDENCE-COSTS not shown (−0.003 [−0.007, +0.000]);
  CRITIC-INFO-LIMITED (the critic search gains only +0.014 [+0.008, +0.021] from the whole truth: a rollout can use it, the
  critic barely).
- **Per class (Amendment 3, F+X worlds at weight p, F at 1 − p):** the seen mons' MOVES rise only at p ≥ 0.75 (+0.013) and never
  beat the critic search (LINEAR); the never-seen mons rise from p ≈ 0.2 (+0.012 → +0.031 at 0.75; P-STAR 0.2 vs the policy,
  0.35 vs the critic search; CONCAVE). Today's p per class in HIT-RATE units (a labelled proxy: no per-class network-belief arm
  exists): moves 0.011, never-seen species **0.28 [0.27, 0.30]** (corrected 2026-10-15 from the published 0.12, which
  counted a species against the truth of the slot it was DRAWN for: the builder-record GIGO, `design_information_levers.md`
  §7.6; recomputed with `unblind_t1.class_dose` on the frozen table, set-based hits; nothing else in Tier 1 reads them); in
  value units the per-class step is again tie-breaking (U +0.007, M none). CLASS-FIRST from that corrected proxy point:
  U-FIRST at k 1–2, k 3 and 12 mons alive (published: NOT DETECTED everywhere); the from-p = 0 verdicts are unchanged.
  CLASS-FIRST (from p = 0): the never-seen mons pay more per unit of their own belief at k ≤ 4 and with 10–11 mons alive.

### 5.2 The value of information by CLASS

| class of hidden fact | registered read (Tier 1, pooled, `design_hidden_info_binding.md` §16.8) | value | tag |
|---|---|---|---|
| **never-seen Pokémon** (identity and sets) | reveal-only share s_U **0.31** [0.18, 0.43], hide-only share n_U **0.82** [0.70, 0.96]; U-LED at k ≤ 4 and turn ≤ 14; R-RIVALS U-SHARED (not a majority by sufficiency) | the LARGEST class; dominates the moves on both sides (R3) | MEASURED, PROVISIONAL; **REPLICATED** at LEAF-K8 on the registered mix (§16.11: s_U 0.31, n_U 0.92 in the blind build on the common states); not on Tier 1's snapshot-game states or end2 alone, in either build; **E2 (§16.12, fresh competent-opponent states, registered): U-FIRST** (s_U 0.40, n_U 0.73), SCORER-DEPENDENT, SHARED per checkpoint |
| **seen mons' unseen moves** | s_M **0.13** [−0.02, 0.25], n_M **0.42** [0.31, 0.54] (E2, competent-opponent states: s_M 0.16 [0.02, 0.28], n_M **0.58** [0.47, 0.70], B4 HOLDS there: a NECESSARY share); never LEADS at any stage or in any E2 frame; the dose helps a little (D90 − F +0.011 [+0.001, +0.021]), a confident wrong set HURTS (W90 − F −0.013 [−0.024, −0.003]), the honest sharpening buys nothing (MAP100 − F −0.002); R-MOVES **REFUTED** | a real minority share | MEASURED, PROVISIONAL |
| **items, abilities, spreads, exact HP** | the fully honest gap ≈ the pilot's (which told them true): IMP_F +0.006 vs IMP_P +0.009, ≈ 0.003; the separate F+I/A/S and O−I/A/S arms are Tier 2 (not run) | small | DESCRIPTIVE (Tier 2 owed) |
| **which TEAM, out of a closed pool** | POOL (the pool's teams consistent with the observation) recovers **0.80** [0.73, 0.87] of G; the pool's per-species set frequencies (POOLSETS) recover −0.02 | large on OUR pool; not available on an open ladder | MEASURED |
| **hidden counters, speed ties** | residual leaks of the honest builder (§6.1) | unknown | UNVERIFIED |

The two-point Shapley credit over the registered 600: φ_U = ½(0.022 + 0.058) = **0.040**, φ_M = ½(0.009 + 0.029) = **0.019** of
G = 0.070; Σ s = 0.45 [0.22, 0.65] (not NON-ADDITIVE), Σ n = 1.24 (the hide-only costs overlap). Nothing from the study may steer
a multi-week programme until the replica agrees, the red team has attacked it and the plateau replication confirms it. The replica
comparison (§16.9, 2026-10-15) agrees on the M row and on G. It is silent on the U row, because the replica built no U arm.
**The blind U-arm build (§16.11, 2026-10-16) fills the U row:** AGREE by its pre-registered criterion on the 978 common
states at LEAF-K8. It also re-reads the M row with the learnset fixed. The moves' hide-only share then differs between the
builds (n_M 0.38 vs 0.54, paired −0.154 [−0.316, −0.005]; B4 holds in the blind build only). s_M stays ≤ 0.13 in both.
The plateau replication is `design_hidden_info_binding.md` §10.5. The author's confidence that the seen mons' MOVES are THE binding constraint is now ≈ 5 %.

**The red team's read of this table (§16.10 of the study; post hoc; the rows above stay the registered read).**
- **The U row is population-dependent.** On the 503 registered states from games against our OWN snapshots:
  - s_U 0.12 vs s_M 0.23, and n_U 0.72 vs n_M **0.58**;
  - R3 is not met, and B4 holds for the moves.
  - On the 697 bot-game states U dominates both sides (+0.38 / +0.57).
- **Per hidden fact** the two classes are worth the same: ν per fact 0.0032 (never-seen, 8 facts per mon) vs 0.0027 (a
  hidden move slot).
- **The items / abilities / spreads row is context-dependent.** They add ≈ 0.0015 alone, but **+0.019 [+0.010, +0.028]**
  once the never-seen mons are known (a complement of U).
- **The R-true two-point Shapley values** are φ_U 0.048 and φ_M 0.020, difference +0.029 [+0.018, +0.039]. The
  interaction is +0.017.
- **Red-team confidence:**

  | statement | red team |
  |---|---|
  | the moves are THE constraint | ≈ 5 % |
  | the moves are a material share | ≈ 85 % |
  | the never-seen class is the largest on the registered mix | ≈ 70 % |
  | … on competent-opponent states | ≈ 35 % |

### 5.3 The honest builder's leaks, fixed 2026-10-12: earlier "fair" numbers were OPTIMISTIC

The pilot's fair worlds (v1, then v2) kept every SEEN opponent mon's item, ability, nature, EVs and IVs TRUE (hence its
exact stats, Hidden Power type and the exact HP its history leaves). Since 2026-10-12 the builder is honest by default
(`WorldSampler(reveal="off")`); the pilot's builder is the `pilot` preset. Truth-match rates (8 bank decisions × 4 worlds,
`info_levers_2026-10-12/audit_8x4.json`):

| fact | pilot's builder | honest builder |
|---|---|---|
| a seen mon's unrevealed item | 1.000 | 0.750 |
| its spread (nature + EVs + IVs, exactly) | 1.000 | 0.032 |
| a seen mon's exact HP at the root | 124 / 124 | 70 / 124 |
| its unrevealed move slots, all right | 0.113 | 0.097 |

**Consequences.** (a) Every pilot-builder IMP (A′, B′, C′, CT′: §2 W5–W8, §3 N9) is an UPPER bound on a fully honest search
(F); §4 and §5 now carry the F numbers. (b) The conclusions that
search does NOT beat the critic are therefore conservative; the late-game edge (W6) is the claim most at risk of shrinking.
(c) The fully honest numbers now exist (Tier 1's F, 2026-10-15): the gap barely moved (G_F +0.070 vs the pilot's
+0.070 / +0.065), so items, abilities and spreads add ≈ 0.003; the pilot's archived worlds and rows are unaffected
(they were built by the pinned twin `determinize.py`), and the honest late edge (W6) shrank to GATE-WEAK at k = 6 (§4).

---

## 6. THE INSTRUMENTS AND THEIR KNOWN LIMITS

### 6.1 The world builder (`src/main/value_search/worlds.py`, `world_draws.py`, `world_screen.py`)

Honest by default: every hidden fact is DRAWN (Smogon prior) and screened against the searched side's byte-equal protocol
and rows; what play revealed (`agents.info_levers.revealed`) stays the truth's; the `oracle` preset reproduces the true
battle exactly (an identity the audit checks). 41 checks PASS on 8 decisions × 6 specs, 12 PASS on 40 decisions
(39 built, 1 excluded); cost ≈ 28 s per decision at 4 worlds on a contended box (the honest screen dominates). **The screen's
cost (MEASURED 2026-10-27, S-U stage 0):** a screening replay is the candidate world's battle from turn 1, so a decision
costs (replays) × (log length) and both climb with the turn (turn 59: 240.5 of 243.0 CPU-s are world building). The
LEAN screen (`WorldSampler(lean_screen)`, the default) stops each replay at its first line that differs from the truth's,
returns only what the gate reads and skips the proposals the prior rules out: the same worlds, draws and verdicts
(`world_screen_parity_test`) UNLESS the core refuses a candidate after its first difference (a check `Err` or a panic
the stop never reaches; 0 in 14,746 replays built to provoke it, and the slow tier now fails on one: adversarial review
[`reviews/lean_world_screen_review_2026-10-27.md`](reviews/lean_world_screen_review_2026-10-27.md), no BLOCKER), 4.3× cheaper over a 66-turn game, 6–8× on its late exclusions. Offline / evaluation only;
no in-Rust builder exists. The pilot's builder cost ≈ 0.08 CPU-s per world (capping in-loop use at f ≈ 0.25 % of decisions); the honest screen is about two orders of magnitude dearer per decision, so in-loop use is farther off (UNVERIFIED, from the two cost figures).

**Known residuals (design_information_levers.md §7.5, §12):**
- **Speed-tie dice (seed reuse).** The prefix is replayed on the TRUE seed, so a hidden fact that spends a die without
  printing a line is visible to the gate. Measured by counterfactual flips: a seen mon's Speed ±1 is rejected for 34 of 674
  (decision, seen mon) pairs (5.0 %), in **20.1 %** of decisions; King's Rock / Focus Band 39 % of tests but 4.7·10⁻⁵ prior
  mass per mon and none in the bank's 231 teams; no unrevealed dice-rolling ability seen. Cure: a playout core that replays on
  fresh dice constrained to printed outcomes (not built).
- **Hidden counters** (sleep / confusion / Encore / lock durations rolled BEFORE the root are the truth's in every world).
  Cure: a root-state resample in the playout core (not built).
- **IVs below 30** (34 % of the bank's Hidden Power sets, 33 % of the others, mostly a minimised Attack) are outside the
  screen's support. **Owner's ruling (2026-10-12): no IV prior below 30**; a minimised Attack is taken as full Attack (it
  matters only in a confusion self-hit or a Struggle endgame), and a decision whose log pins the Attack stays EXCLUDED
  (counted), never misbuilt.
- **The factorisation** (each seen mon screened with the others at their true facts), the **approximate** posterior chain
  (Metropolis-Hastings after a short burn-in), the **spread prior** (a Smogon table plus a uniform floor), and **party-wide
  effects** (Beat Up) are measured by rejection counts or named, not proven.
- **The construction-draw gender coin:** the engine pad (`98146ddb`) removes the pilot v2's residual conditioning
  (accepted P(m = r) 0.67 / 0.64 vs the ungated proposal's 0.57); a never-seen ratio'd mon on a true draw slot still takes the
  truth's gender coin (unobservable before the root).
- **The independent re-implementation:** the blind replica's findings were relayed and answered (gender dice, dice-silent
  facts, `playout_open`). Its third-party comparison was READ on 2026-10-15 (study §16.9).
  - **What agrees:** the truth, the referee and the scorer are identical, and the searched side's view is equal on 1,164 /
    1,164 states.
  - **Common-mode parts it CANNOT check:** the same Phase 1 rows, the reuse of the true dice seed (the dice-silent leak is in
    BOTH builders), and the same written spec.
  - **Its own handicap:** the replica's move prior used the learnset table before `64a695dc`. That inflates its move shares,
    bounded at n_M ≤ +0.10. **FIXED in the blind U-arm build (§16.11):** its missing-pair hit rate is 0.313 vs the
    original's 0.321. Yet ν_M still differs by −0.0078 (−0.0067 with the handicap), so the handicap was NOT the cause.
    The seen mons' move and nature draws still differ beyond a permutation null. Part of the move difference is Hidden
    Power's presence in drawn sets (the spec's open "HP type: M or S" question; UNVERIFIED as the cause).
  - **The never-seen arms now have two builds (§16.11).** The instrument is identical: IMP_O, Phase 1, one scorer, and the
    view on 978 / 978 states. U's drawn content (species, item, ability, nature) is at the permutation null. U.move differs
    slightly. The first never-seen slot's species differs by construction: uniform vs slot-ordered placement, the red team's
    E7, and the U contrasts agree anyway. The blind guard's one false positive is an untyped Hidden Power in one true set; it
    was ruled an exclusion, not an identity failure.
  - **§12.2's TV < 0.02 test is unmeetable at this sample size:** two halves of one builder fail it.
  - **The honest builders' exclusions are not missing at random.** On the states only the replica could fit, n_M is 1.17.
- **ILLEGAL worlds (MEASURED 2026-10-19, FIXED the same day; `design_world_sampler.md` §6.1, §7 F2).** Showdown's
  `TeamValidator('gen3ou')` rejected **581 of 11,160** stored F worlds (5.2 %), in 445 of 1,395 states (32 %); the true
  teams: 0 / 1,395 fail. The causes were egg- and event-move COMBINATIONS (Tyranitar Pursuit + Dragon Dance, Skarmory
  Whirlwind + Drill Peck, Blissey Wish + Aromatherapy) that per-slot move draws cannot see. **The builder now draws
  every set from the prior conditioned on the validator's acceptance** (rejection; the truth must pass or the build
  raises): on 316 random states rebuilt, 0 illegal worlds, no exclusion caused by the rule, and the species marginal
  unmoved. **Every honest read so far (Tier 1, E2, the maturity battery at its pins) used the illegal-capable builder;
  its effect on IMP, read by stage 1, is NOT DETECTED:** the rule alone (F_legal − F_off) +0.005 [−0.001, +0.011], every
  builder change since the banked F +0.003 [−0.007, +0.013]. Worlds built before 2026-10-19 are not comparable with
  worlds built after (the rule and the same day's Hidden Power fill).
- **The true-dice replay over-sharpens spreads (MEASURED and PARTLY RESOLVED 2026-10-19, `design_world_sampler.md`
  §3.3.1, F1).** A world had to reproduce each observed damage number with the truth's own damage ROLL, where an honest
  player knows only the 16-roll range. MEASURED: 9.0 % of the candidates today's screen rejects fit on another roll.
  **Tier A is BUILT behind `WorldSampler(dice="rolls")`, default OFF**: the damage roll's VALUE replaced within the
  aligned stream, weight Π k / 16, the truth-passes guard raising. On 60 rebuilt states: exclusions 20 → 6, cost
  96 → 42 s per state, item / ability / spread matches with the truth 0.953 / 0.989 / 0.056 → 0.951 / 0.980 / 0.047,
  weight ESS 7.3 of 8. **Every honest read so far used `seed`;** worlds of the two modes are not comparable. Tier B
  (every draw site typed: the crit roll, speed ties, silent dice, the counters) is not built. The search consumer
  reads the weights on request since 2026-10-19 (`SearchConfig(weights="likelihood")`, uniform allocation only). Stage 1
  read tier A both ways: uniform +0.003 [−0.012, +0.018], weighted −0.003 over F (NOT DETECTED); it is a correctness
  change (exclusions 7 → 4 of 287), not a value lever.
- **Tier 1's POOL arm at HEAD is not the pinned pool** (FOUND 2026-10-19, `design_world_sampler.md` §7 F9): the Hidden
  Power refactor re-packs 34 of the 719 pool teams with other IV patterns, so a POOL read on a pinned bank must load the
  pin's packing (stage 1 does: `pool_pin.py`).
- **What the exclusions are (2026-10-19).** On the maturity LEAN and E2 banks, 125 of 1,520 states are excluded. 121 of
  them are a seen mon whose item / ability / spread screen found no fit in 200 chain steps (Tyranitar 23, Zapdos 17,
  Swampert 14, Jirachi 13, Metagross 12), and 4 had too few worlds. A live search cannot exclude a decision
  (`design_world_sampler.md` §3.5, owner Q2).
- **The world records (FIXED 2026-10-15):** `Worlds.facts` kept a never-seen mon's record at the slot it was DRAWN for
  after the gender placement permuted the mons (46 % of the stored Tier-1 never-seen records at another mon's slot). The
  WORLDS were right (whole entries moved), so no playout and no registered verdict changed; the records and the
  set-vs-slot species hit did. Every returned world is now checked slot ↔ species ↔ facts ↔ the ENGINE's view and a
  mismatch raises (`world_guard`, `design_information_levers.md` §7.6). It also caught 7 POOLSETS worlds (of 9,112)
  with Will-O-Wisp twice: one pool team packs the move as a Showdown name variant, `wisp`, and the packed-set reader
  compared the raw token (FIXED 2026-10-15, one reader + throwing guards, §7.7). Past reads: 0 states with a variant in the
  TRUE team, so F / FN / O−* / D* / W90 / MAP100 and POOL are unmoved; the exact leave-one-out of the 7 POOLSETS worlds
  moves the arm-mean IMP by ≤ 3.3e-4 (r_POOLSETS −0.02 [−0.19, +0.13]); stamped, not re-run.

### 6.2 The IMP metric and its disjoint referee

IMP(scorer) = T(scorer's pick) − T(the policy's top-1) on the win scale, where T is the mean outcome of 24 truth playouts of
that action (seeds 16–39) that **no scorer reads** (leaf seeds 0–7/0–15, critic 40–47): the pick cannot be rewarded for fitting
its own dice. States: the 600 per checkpoint from its own eval traces (loss-enriched, stratified by the policy's top-2 gap as
a proportional sample), all legal actions, battle-clustered bootstrap (2,000 draws, seed 20261011). Limits: a one-step policy
improvement against a GREEDY continuation (not Nash, not an adaptive opponent); **a per-decision improvement is not a per-game
win rate** (never measured: S0b has no ledger entry); the raw IMP is lowered ~10–20 % when re-weighted to the natural outcome rate; the
winner's-curse ORACLE upper bound is +0.152 / +0.132; off-distribution states for any other checkpoint; the 15M policies
were not plateaued. **The referee plays the searched network on BOTH sides** (the self-referee): every bot-stratum IMP here
is a gain against our own continuation. `main.value_search.playout.referee` (2026-10-25) reproduces it exactly and takes
`opp_bot` (default OFF) to play the opponent as the state's in-core scripted bot; the search's leaves (`LeafEngine`) take
no bot. E1-0 (the status line) is the first read with it: UNDECIDED at 160 states, the gains shrinking against the bot.

### 6.3 The information-lever layer (`agents.info_levers`; `design_information_levers.md`)

ONE declarative spec (seven classes of hidden fact × seen / unseen, a mass and a source per lever: truth, a
wrong-but-plausible placebo, an honest sharpening) read by three consumers: the **search world builder**, the **training
observation** (`--oracle-reveal <spec>`, with `--diagnostic-fork-consent` for a fork; BUILT), and the **model-input
injection**. Presets cover the X32 levels, the oracle, the pilot's builder and every hidden-info study arm; a protocol-derived
log source says what play revealed; the audit suite is `python -m main.info_audit`. How to read ceilings with it: oracle =
the information ceiling, `off` = today, the per-class gap (reveal only X / hide only X) = what improving that class could buy
at most, the dose = how much a partly better guess buys, the placebo = content vs confidence. Limits: the observation cannot
state an exact HP, a hidden duration, a placebo or a map fact (the knock-in forks' wrong-moves arm needs a prior sampler on the
Rust side, not built); the training-time `--belief-oracle` of the study's §8.1 is NOT built. The knock-in forks ran on the
observation route (§7 Q3), so the reveal reached the actor and the critic alike; a critic-only route does not exist (the extractor
feeds one shared trunk to both heads).

### 6.4 The live searched player (`src/main/anchors/search_side.py`; `main.anchors --our-search`; 2026-10-27)

The first player that searches INSIDE a game against a non-self opponent (S-U's B1–B3, `design_search_u_headroom.md`
§6). P\* plus R0 §4's search (`r0s4`: 8 honest worlds `dice="seed"`, TRUNC-4-K8, piKL λ 0.015, now
`SearchConfig(rule="pikl")`, which re-scores L1's banked rows exactly) at our decisions from turn 10 with ≥ 2 legal
actions, as the in-process core slot against a local Metamon peer; every other decision is P\*'s greedy choice. Its
limits:
- **The TRUE log, in process** (`design_search_next.md` D-8). The front end records every command with the request
  update it answers; the search replays every update before ours, never the opponent's choice of our own update (its
  root decision is open and π answers it). A guard per decision: the log's replay must reproduce the live decision's
  tokens and observation row byte for byte, or the game fails by name. **The leaks of §6.1 stay** (the true seed's speed
  ties and hidden counters): a live gain is LEAK-INCLUSIVE.
- **Paired by deal.** Raw and searched games of one deal (a seed's game G, `main.anchors --deal-game`) must be
  byte-identical through `|turn|10` (G-PREFIX, `main.reads.prefix`); a mismatch holds the lane and voids the group.
- **The peer's patience** (FINDING S-3, closed): the only clock on Kakuna's side is poke-env's websocket keepalive
  (20 s + 20 s), so the search runs off the event loop; a 70-s off-loop decision kept Kakuna connected
  (`src/main/anchors/peer_patience_integration_test.py`). The negative control, the same 70 s ON the loop, dropped the
  peer (`keepalive ping timeout`) and the front end booked a forfeit WIN for us at turn 3. The forfeit-win guard
  (2026-10-27, `EXTERNAL_ANCHORS_SOP.md` § Counting a unit) now voids any game a forced end decided other than our own
  turn-limit cap, in every anchor lane; the same control is its slow test.
- **Cost.** The build's smoke (12 searched decisions, turns 10–19) measured 6.4 CPU-s a decision; **it does not hold
  past turn 20**. S-U stage 0's first searched game (home72, 66 turns) climbed from ≈ 6 CPU-s a decision at turns
  10–19 to ≈ 130 at turns 50–60 and up to 470 for a late exclusion: 77.8 search CPU-min, with worlds, depth, tries and
  playouts flat or falling. **MEASURED (2026-10-27): it is the honest world screen** (§6.1; the decision log now splits
  `world_cpu_s` / `playout_cpu_s`). On the lean screen the same 57 decisions are made identically for 18.0 CPU-min.
  Spread over X51's game lengths that is median ≈ 5 CPU-min a searched game, mean ≈ 12–16, the mean set by the long
  games (ESTIMATE, ONE game; `../research_state/measurements/su_headroom_2026-10-27/AMENDMENT_1_DRAFT.md` §1b). In the smoke all 12 decisions from turn 10 had ≥ 2 legal actions; 1 of 12 deviated from π. The same
  searched deal played twice was byte-identical over the whole game, decision records included; G-PREFIX held on both
  pairs, and the home72 pair diverged at turn 11, where the search first deviated.
- **Offline at kept states (the S-U gate test, 2026-10-27).** A RAW cell can keep each decision's honest input log
  (`main.anchors --keep-input-logs`, `src/main/anchors/input_log.py`); `main.value_search.sign_gate` re-runs the live
  searcher's own `_search` there (its keying and root guards: the kept log must replay to the live tokens and row) and
  scores S's deviations with the self-referee (`playout.referee`, 24 seeds, common dice). Rule `deviation_sign`.
- **The rule** `search_headroom` (B4, `src/main/reads/rules_search_headroom.py`) decides S-U's classes. S-U is REGISTERED
  (`../research_state/measurements/su_headroom_2026-10-27/PREREG.md`, pin `bf5f1a8e7`); no full live read has been
  played.

---

## 7. OPEN QUESTIONS, and the experiment that settles each

| # | question | the experiment | cost / status |
|---|---|---|---|
| Q1 | Does a FULLY honest search beat the critic anywhere, and which class of fact carries the gap? | **Tier 1 of the hidden-info study** — READ 2026-10-15 (§16.8): no (FRONTIER EMPTY; k = 6 GATE-WEAK); the never-seen Pokémon (R-MOVES REFUTED by R3). **Tier 2** (F+I/A/S, O−I/A/S, D25 / D75, W50 / W100, MAP90, P16 = R-WIDTH, the second referee) is the open remainder | Tier 1 cost ≈ 1.5 days on 3 agent cores (FN built; the honest screen ≈ 20–60 CPU-s per state). Tier 2 not queued |
| Q2 | Is the pilot's oracle − honest gap an artifact of our instrument? | **The blind replica's third-party comparison** (R-REPLICATE, then the red team, `design_hidden_info_binding.md` §13) | **PARTLY ANSWERED 2026-10-15 (§16.9):** the gap is NOT an artifact of one builder (G replicates on the same states). R-REPLICATE as registered is NOT MET: the ν_M point rule fails, and TV < 0.02 is below the noise floor. R3's never-seen half is unreplicated. **The RED TEAM has run (§16.10):** the instrument holds against every attack it could make offline (tie-free scoring, the exclusions, the mis-keyed records, a U-leak check, the U arms' team text). What it cannot settle offline is the never-seen half: the blind U-arm build E1 (16–40 CPU-h, ready to dispatch in `red_team.md` §3), the competent-opponent bank E2 (40–65 CPU-h) and the second referee E3 (3–5 CPU-h). **E1 READ 2026-10-16 (§16.11):** AGREE by its pre-registered criterion at LEAF-K8 on the 978 common states, so R3's never-seen half is REPLICATED, scoped to the 8-rollout leaf and the bot-heavy registered mix (both builds fail it on snapshot-game states and on end2 alone). R-REPLICATE §12.2 as a whole is still NOT MET on the M side. **E2 READ 2026-10-18 (§16.12):** U-FIRST on 1,200 fresh competent-opponent states by the registered pooled LEAF-K8 primary, SCORER-DEPENDENT and SHARED per checkpoint; the GIGO audit is clean. Open: E3 (the second referee), and the plateau replication (Q4) |
| Q3 | Can a network LEARN to use told information, and what is each class worth in strength? | **The knock-in forks**: `--oracle-reveal ki_moves` / `ki_items` / `ki_spreads` / `species:unseen@0.95` with `--diagnostic-fork-consent`, 5 arms × 3 seeds × +3M from the adopted end state, a no-change control (the v8 guard); no placebo (the observation refuses one) | **ANSWERED at +3M, PROVISIONAL (2026-10-15, `design_hidden_info_binding.md` §8.7):** it learns to READ moves and species (withheld costs ≈ 1.9 pp) but GAINS nothing in strength (moves, items, species NULL; spreads NOT DETECTED). Open: whether longer exposure, a critic-only route or an annealed reveal leaves a durable honest gain; which mechanism carries the dependence (§8.7's experiments (a)–(e), none queued) |
| Q4 | What do maturity and plateau change: does G shrink, does today's p rise toward p\*, does the frontier move earlier? | **The maturity battery** on the X26 deep run (`design_hidden_info_binding.md` §12.13.7): O, F, FN, F+M, F+U re-run at ≥ 25M then every 25M and at the plateau checkpoint, ONE frozen 800-state bank, rules R-MATURITY (GAP-SHRINKS / BELIEF-RISES / FRONTIER-MOVES-EARLIER / -STATIC / -RECEDES). **Harness BUILT 2026-10-18 (T35)**; **the first trajectory REGISTERED 2026-10-18**: an INTERIM 800-state bank (LEAN 320; half own-snapshot states) from X26's 230M–234M cycles, LEAN at 25M / 100M / 175M / 225M, the plateau bank to supplement it (`measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md`). Measured LEAN ≈ 7–8 CPU-h per anchor (not 14); **the first LEAN trajectory READ 2026-10-19** (README §5; PROVISIONAL, interim bank): GAP-SHRINKS NOT DETECTED (G_F +0.075 → +0.053, change −0.022 [−0.049, +0.004]), BELIEF-RISES NOT DETECTED (TODAY ≈ 0.01), NOT CONVERGED (unreachable at the registered sizes: a FINDING), FRONTIER NOT READ (LEAN), every gate CLOSED. **The PLATEAU STEP REGISTERED 2026-10-19** (amendment A1, PREREGISTRATION §9; README §6) **and READ 2026-10-25** (README §7): the record, the plateau bank's [25M, P\*] on 719 states, reads GAP-SHRINKS NOT DETECTED (−0.003 [−0.020, +0.015]), BELIEF-RISES NOT DETECTED, FRONTIER-STATIC, STILL-MOVING NOT DETECTED, CONVERGED NOT READ (pooled needs 225M on the plateau bank); the interim replication reads GAP-SHRINKS (−0.030), not BANK-DEPENDENT; R3 at P\* SHARED on both banks. Every cell SELF-REFEREE (E1-0); a bot-referee re-score is proposed. The code: `src/main/hidden_info_study/maturity_{series,bank,rules}.py` (the 25M trigger, the read-only checkpoint copy, the frozen bank sampler, R-MATURITY as code), drivers and queue in `measurements/hidden_info_maturity_x26_2026-10-18/` | ≈ 150 CPU-h for six checkpoints (FULL ≈ 47 at the first and plateau anchors, LEAN ≈ 14 between), no GPU. The plateau step has run (2026-10-25) |
| Q5 | Does the honest ONE-PLY critic search pay in GAMES? | **S0b** (`design_value_discrimination.md` §13.4): the end-state checkpoint with an 8-world one-ply critic search vs the same checkpoint raw, mirrored head-to-head; pass = lower bound > the +4.57 pp replicate floor | ≈ 1 CPU-day. NOT run as far as the ledger records; OWNER approval requested (§13.0 ask 2) |
| Q6 | Is the stage-gated bootstrap viable: does honest search work anywhere well enough to label? | **R-STAGE-SEARCH** (Tier 1S, 13 buckets, per checkpoint and pooled at 99.6 %), then the S1 SHADOW (logs only, per-stage `search/imp`), then the pre-registered A/B (`design_value_discrimination.md` §13.4, §13.7, §13.10) | A DESIGN PROPOSAL for the owner's review, nothing built. Gate closed today (§4). The in-training deferral (2026-09-30) is NOT lifted by this doc |
| Q7 | The oracle-critic roles: may an oracle-informed critic or oracle labels help? | (a) the offline omniscient twin X11 as a measuring instrument (V − E_belief[V\*]; where is error reducible, where is information worth gathering), never a target; (b) oracle playouts as value labels for a TRAINED body or a much larger label budget (`design_value_discrimination.md` §13.8, GPU; needs the owner's ruling on in-training search). The 09-10 `truevalue` arm (N7) argues against training the public V on true values | not run; (b) is the only route that reaches the hidden-information-limited label |
| Q8 | Is the move-belief gap closable at all? | ANSWERED for search at 15M (Tier 1, 2026-10-15): the moves are a minority share (N14) and the network's belief worlds buy nothing (N15); today's belief sits below the tie-break step (§5.1) | for the NETWORK: the knock-in forks (Q3) read NULL at +3M. Even the told truth does not make the policy stronger, so closing the move-belief gap is not a policy lever at this maturity |
| Q9 | Is the evaluator or the information the limit (Foul Play's hand evaluator vs our critic)? | **R-EVAL** — READ 2026-10-15: not EVALUATOR-BOUND, INFORMATION-BOUND support (N13); width (Foul Play's other lever, R-WIDTH / P16) is Tier 2 | width open |
| Q10 | Does the full-length leaf need the per-turn reseed (S1)? | R-PAIR said ALIGNMENT-LOSS; S1 (~0.5–1 agent-day) matters only if a full leaf is still wanted after the 4-turn leaf | not built; conditional |
| Q11 | What does one-ply best-response give up against an equilibrium method? | the cheapest equilibrium-flavoured step: an MMD-style magnet term (`design_value_discrimination.md` §14.3), or the per-turn matrix-game solve (§14.4). Its gen-3 form: each sampled world is an opponent TYPE, one mixed strategy for us and a best reply per world, a small linear program. That form, its harness changes (a FORCED opponent root action; a depth-1 version on `Successors.expand_many` with no Rust change) and a best-response referee are in `design_world_sampler.md` §5 | not built, not registered; ranked AFTER the sampler (a matrix over worlds worth 1 % of the truth solves the wrong game) |
| Q12 | Which world sampler puts search-sized weight on the truth, and do partly right worlds carry value? | **Stage 0** (zero playouts, READ 2026-10-19): the network holds the species, nothing holds the sets (§1). **Stage 1**: the sampler ablation on the maturity LEAN 287 states at X26 225M (arms: joint Smogon teammates, a team-level corpus model on the forum corpus minus near copies of the truth and on the ladder corpus, the pool ceiling, the network's species × the forum's sets), with PAYS / HURTS / NULL / MATERIAL / SEARCH-ENABLED rules (`design_world_sampler.md` §6.2) | **READ 2026-10-21** (`measurements/world_sampler_stage1_2026-10-19/`): only POOL pays (r 0.72); every open-population sampler NOT DETECTED; partly right worlds carry no value at LEAF-K8 (§1, N17). The 1b trigger fired on POOL alone; **stage 1b NOT run** (orchestrator, 2026-10-21: the closed-pool ceiling is not available against real opponents) |
| Q13 | Does honest search pay once its acting rule is ANCHORED to the policy (the optimizer's curse; Ataraxos's anchor)? | **L1** of the literature review: piKL / Gumbel scale / a deviation threshold, re-scored on the banked honest rows; λ fitted on Tier 1, read held out | **ANSWERED 2026-10-20** (pre-registered `d443c098`): PAYS-SCOPED on one cell (piKL, E2, TRUNC-4-K8 +0.012, lower +0.001 at 98.33 %), which is the leaf's own E2 signal (the anchor adds +0.002 [−0.005, +0.008]). The anchor removes the argmax's harm on X26 (−0.012 → ≈ 0) and beats the policy nowhere; the sampler binds, not the acting rule. **Replication READ 2026-10-25** (plateau bank, `../research_state/measurements/search_l1_plateau_2026-10-25/`, SELF-REFEREE, Tier-1 fit frozen): the cell REPLICATES, +0.0097 [+0.0025, +0.0169], 98.33 % lower +0.0010; the anchor's own part +0.0025 [−0.0039, +0.0089] (not clear); the gain is at the 25M anchor (+0.0193), none at P\* (+0.0001). Still the leaf's signal at an immature policy, not the anchor's. Nothing open on L1 |
| Q14 | Does search need weight on the exact truth, or a CALIBRATED belief? | **L2** (a descriptive amendment to stage 1 before its read: IMP against the truth's log-likelihood, not q_SM) and **L4** (a bank whose opponents' teams are drawn by F's own sampler) | L2 zero; L4 ≈ 1–2 agent-days + 10–20 CPU-h; neither registered |
| Q15 | Are the opponent's own past choices usable evidence about its hidden MOVES? | **L3**: the truth's rank among honest worlds by the likelihood of the opponent's observed choices under its policy, per opponent stratum | **ANSWERED 2026-10-20** (pre-registered `4d149799`): YES as a judge, PROVISIONAL-POOL. The truth out-ranks honest worlds, AUC 0.692 [0.655, 0.728] (WORKS), carried by the snapshot stratum (0.796; bots 0.588, WEAK alone); the confidence control reads 0.481, but pool familiarity is not ruled out by it (review F2). The familiarity counter-test (pre-registered, NOT-DETECTED): pool decoys beat honest worlds (0.575, clear), and the truth beats its decoys only inconclusively (0.578 [0.489, 0.664]), so part of L3 is pool-ness. **Off-pool at P\* (`search_offpool_bank_2026-10-20/`): REPLICATED-SELF-PLAY-ONLY** (snapshots 0.794, bots 0.472 NO-EVIDENCE), and an off-pool ladder decoy gets no lift (0.503): the pool lift was memory, not coherence; the truth beats its off-pool decoys (0.611). Re-weighting today's 8 honest worlds barely moves the moves (+0.005 pooled), so the next step is stage 1c with a bigger or team-level proposal set. **Stage 1c's PHASE A READ 2026-10-20** (`../research_state/measurements/search_1c_2026-10-20/`; 128 F + 128 C_ladder proposals per state weighted by dice × the likelihood): NEARER on all three primaries (ΔH +0.007 F, +0.019 corpus STRICT, +0.013 both), none MATERIAL, PROVISIONAL-POOL and SELF-PLAY-ONLY (bots NOT-DETECTED); the likelihood's own selection +0.008 on F (snapshot-only), not detected on the corpus; the exact-world mass ≈ 0 (≤ 0.0003). **Phase B READ 2026-10-21:** honest search over the resampled worlds with L1's anchor is NOT-DETECTED against the policy on every arm (C128L +0.007 [−0.003, +0.017], n 285; Holm: none pays): NEARER BUT NOT PAYING. Open: the same screen on E2, an off-pool bot bank rich in P\*'s losses, phase A / B on the off-pool bank (only the snapshot stratum can carry a gain there), and a HUMAN OPPONENT MODEL for the ladder (our network as the opponent model shows nothing against a non-self opponent off-pool). **That model is DESIGNED + PRE-REGISTERED 2026-10-23** ([`design_x50_rung1b_human_opponent_model.md`](design_x50_rung1b_human_opponent_model.md)): X26's network fine-tuned from P\* on ≈ 2M human (≥ 1500) ladder choices, the viewer's own side completed from Smogon sets, used ONLY as this likelihood's judge, never read by the policy. Decisive read R-b2: the off-pool bank re-scored with it as the judge; PASS (HOM-EVIDENCE) = the bot stratum WORKS (lower > 0.555) AND above P\* paired, which re-opens this route for non-self opponents (stage 1c on the off-pool bank); HOM-NONE closes it and leaves public-team proposals. ≈ 15–18 CPU-h, 0 GPU-h; the owner approved it 2026-10-23. **R-b2 READ 2026-10-24: HOM-NONE** (`../research_state/measurements/x50_r1b_hom_2026-10-23/`): the HOM is LEARNED on held-out humans (R-b1: −0.582 nats [−0.599, −0.565] against P\*), yet as the judge on the off-pool bank its bot stratum reads 0.459 [0.406, 0.510] (NO-EVIDENCE) and its paired ΔAUC against P\* −0.013 [−0.068, +0.045]; snapshots fall to 0.706 (P\* 0.794). The lever is CLOSED for search at P\*; the bot proxy (F-1b-3) leaves the ladder's own read (L3-H) unbuilt |
| Q16 | Does anchored one-ply search ADD to an exploiter, or DEFEND its target against one (the robustness side)? | **R0's search cells** (X35 `R0_REGISTRATION.md` §4, registered 2026-10-20): L1's piKL (λ 0.015, TRUNC-4-K8, 8 honest worlds) on a best responder's final against its target, and the target + piKL against the best responder; 100 paired deals a cell, DESCRIPTIVE. Predicted: +1 pp added to the exploiter, +3 pp of defence (by unpredictability, not better play). L1's decision-level IMP cannot answer it (its referee plays the searched network on both sides) | ≈ 53 CPU-h a cell, 4 cells; needs a LIVE one-ply search player in games (not built; S0b's was never built) |
| Q17 | Does one-ply search on P\* gain MORE where P\* pilots an unfamiliar team (U) than on its own (home72), and does it close goal (b)'s gap? | **S-U** (`design_search_u_headroom.md`, DRAFT): P\* vs P\* + piKL from turn 10 against Kakuna, home72 and pooled U, one engine, paired by deal; U-HEADROOM / U-LESS / NO-U-PREMIUM on the interaction, DOES-NOT-CLOSE-B on G_U (positives leak-inclusive). H-MID (2026-10-26) does not rule it out and does not motivate it (review M4: the search leaf is the self-play regime, where the premium cancels); S-U is an independent question. Its within-node premise (F5 of the 10-27 survey) is V-RANK's (`design_v_rank_on_u.md`, DRAFT), a filter that can turn S-U OFF | ≈ 560 CPU-h central (480–640) at 1,200 deals a set on the lean world screen (ESTIMATE, one game; the old builder ≈ 2,000–3,000); the live search player is BUILT (§6.4); a rented-box read; registered, stage 0 HELD. V-RANK ≈ 96 CPU-h, not built |

---

## 8. THE MAINTENANCE RULE

1. **Every search-related result updates this ledger in the same commit**: a new IMP, a stage cell, a gate verdict, a builder
   change that moves a leak, a replica or red-team finding, a fork read, a maturity checkpoint. Add or change the row, move
   the claim between §2 and §3 if its verdict flips, and say what changed in the Decision record.
2. **Claims cite ledger entries** (`research_state/ledger.md`, found through `ledger_index.md`) or the design section that owns
   the number; this doc never carries a number its owner does not. The ledger wins any disagreement; fix the view, never the ledger.
3. **Pre-plateau results are labelled PROVISIONAL**; they stay so until the maturity battery (Q4) has run at the plateau
   checkpoint (it has, 2026-10-25, for its own reads; lifting each other claim's label is decided per claim, not yet done). A pre-Rustboro result is labelled PRE-ERA, kept as history, and paired with its current-era re-measure.
4. **A post-hoc read stays DESCRIPTIVE** here even when it is large; only a registered rule's verdict (and its replication
   per checkpoint) moves a claim to MEASURED-and-steering.
5. **The honest-builder caveat travels with every honest number** (§5.3) until the fully honest base (F) has been read.
6. **A reported hazard is a FINDING**: a leak or a defect in an instrument goes in §6 with its measured size or "UNVERIFIED".
7. This doc is read before reasoning about search (root `CLAUDE.md`, "Where the detail is") and linked from
   `research_state/UNDERSTANDING.md` §4.7.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-12 | **This doc CREATED at the owner's request** ("try and build a ledger of what we know about search, what we know about where it's working, where it's not working, how much we need to improve for it to work … establish the ledger now") | ONE always-current index-and-verdict doc: a one-screen summary; what works / what does not (each with an evidence tag, its interval and a pointer); the FRONTIER MAP by stage (reveal count, turn, MATERIAL); how far we are (the belief-quality dose-response against today's effective p, the per-class value of information, and the honest builder's 2026-10-12 leaks); the instruments and their limits; the open questions each with its settling experiment; a maintenance rule that every search result updates it in the same commit. It holds no measurement of its own | scattering the verdict across `design_value_discrimination.md`, `design_hidden_info_binding.md` and `UNDERSTANDING.md` §4.7 (no one place says where search stands); folding it into `UNDERSTANDING.md` (that file is the research view; a search-specific frontier table and the instrument limits would swamp it); reading the blind replica's code or results to fill Q2 (the owner's rule: its comparison is third-party and pending) | `design_value_discrimination.md` §7.11–§7.14, §13.10; `design_hidden_info_binding.md` §12.13, §16; `design_information_levers.md` §7, §10; `measurements/value_disc_pilot_2026-10-11/`, `hidden_info_binding_2026-10-11/`, `hidden_info_by_reveal_count_2026-10-12/`, `info_levers_2026-10-12/`; `UNDERSTANDING.md` §3.1, §4.5, §4.7, §4.x; ledger 2026-10-11 L24263 / L24341 / L24381 / L24485 / L24511, 2026-10-12 L24578 / L24604 / L24663 |
| 2026-10-15 | **TIER 1 READ (the hidden-info study's registered factorial on the FULLY honest base)** | §1, §2 (W9 qualified, W12 POOL), §3 (N13 hand evaluator current era, N14 R-MOVES REFUTED, N15 FN), §4 (the registered Tier-1S frontier map replaces the pilot's descriptive one: FRONTIER EMPTY, U-LED early), §5.1 (the dose on F with the POST-HOC tie-break FINDING: the step to p = 0.1 is reached at p = 0.001; Amendment 3's per-class curves), §5.2 (class shares), §5.3 (c), §7 Q1 / Q8 / Q9 updated | keeping the pilot's descriptive frontier as the map (it ran on the leakier builder; it stays in its owners); quoting the registered P-STAR = 0.1 / TODAY = 0.009 without the tie-break qualifier (a 10 % belief would look sufficient) | `design_hidden_info_binding.md` §16.8; `hidden_info_tier1_2026-10-13/` README §7, `result_t1.md`, `posthoc_tiebreak.json`; ledger 2026-10-15 |
| 2026-10-15 | **REPLICA COMPARISON READ (a third party; the study's §16.9)** | Status line; §1 (G REPLICATED; the moves' shares replicated, U's dominance single-implementation; the open list now leads with an independent F+U / O−U); W9 (the tie-break step replicates on the replica's own worlds); §4 header (the stage table NOT replicated, and the replica's k-gradient sub-read differs); §5.2 note; §6.1 (what the comparison can and cannot check: common-mode parts, the replica's learnset handicap, §12.2's unmeetable TV threshold, the not-missing-at-random exclusions); §7 Q2 PARTLY ANSWERED | marking R3 / U as replicated because the M half agrees (the replica built no U arm); dropping the "computed" caveats wholesale (only G and the M shares gained a second implementation) | `design_hidden_info_binding.md` §16.9; `hidden_info_tier1_2026-10-13/replica_comparison.md` / `.json`; ledger 2026-10-15 |
| 2026-10-15 | **GIGO FIX: the world builder's never-seen fact RECORDS were paired with the wrong slot; the consistency guard built** | §3 N15 (the FN species hit 0.411 vs 0.180, set-based), §5.1 (Amendment 3's U TODAY proxy 0.28, CLASS-FIRST from it), §6.1 (the records residual, fixed) | re-reading any playout (the worlds were right: whole entries moved; every IMP, G, s / n, R-STAGE, R-DOSE and R-RECOVER number stands) | `design_information_levers.md` §7.6; the impact table and the recomputation: ledger 2026-10-15 (GIGO FIX, world records) |
| 2026-10-15 | **RED TEAM READ (the study's §16.10)** | Status line; §1 (the never-seen bullet WEAKENED: population-dependent, not met on end2 alone, equal per hidden fact; the "how far" bullet: "0.2–0.35" is the leaf's quantisation, FN's species belief 0.41 vs 0.18; the open list re-ordered by the red team's settling experiments); W9 (the quantisation qualifier); N13 (R-EVAL a near miss toward evaluator-bound); N15 (the species hit CORRECTED to 0.411 / 0.180, `8702a56f`; FN − F at TRUNC-4-K8 a boundary); §5.1 (the continuous-leaf curve); §5.2 (the red team's read and confidence); Q2 | rewriting the registered rows (they stay the registered read; the red team's numbers sit beside them, labelled); dropping "0.2–0.35" without the replacement curve (the continuous leaf's curve is the honest replacement) | `design_hidden_info_binding.md` §16.10; `hidden_info_tier1_2026-10-13/red_team.md` / `.json`, `red_team.py`; ledger 2026-10-15 |
| 2026-10-15 | **THE KNOCK-IN FORKS READ (the study's §8.7)** | Status line; §1 (a new bullet: information is a SEARCH lever, not a POLICY lever, at this maturity; "what does not help" gains the trained reveal; the open list drops the forks' read); N8 (the forks' numbers and the reading: the network learns to READ the told moves, not to gain from them); §6.3 (the forks' route reached the actor; no critic-only route exists); Q3 ANSWERED at +3M; Q8 | moving N8 into §2 (no reveal bought strength); quoting the honest dependence cost as a verdict (descriptive, no rule); quoting the v8 +4.4 pp here (it is a training-maturity read, not a search one, and CONFOUNDED) | `design_hidden_info_binding.md` §8.7; `measurements/hidden_info_forks_2026-10-14/` (`crosses/result/verdict.json`); ledger 2026-10-15 MEASUREMENT (MAJOR) "THE KNOCK-IN FORKS" |
| 2026-10-16 | **THE BLIND U-ARM BUILD (E1) COMPARED by a third party (the study's §16.11)** | Status line; §1 (the never-seen bullet: U's dominance REPLICATED at LEAF-K8 on the registered mix; the population split and the end2 failure replicate in both builds; the third party's confidence; the open list: E1 done, E2 decisive); §5.2 (the U row's tag and the M row's re-read with the learnset fixed); §6.1 (the handicap fixed but the M-side difference persists; the never-seen arms' two builds); §7 Q2 | marking R3 replicated without its scope (it is a LEAF-K8, registered-mix claim; it flips at LEAF-K4 / K1 and fails on snapshot-game states and end2 in both builds); dropping §16.9's handicap attribution silently (it is stated as not supported, with the numbers) | `design_hidden_info_binding.md` §16.11; `measurements/hidden_info_u_replica_2026-10-15/third_party_comparison.md` / `.json`, `third_party_compare.py`; ledger 2026-10-16 MEASUREMENT (MAJOR) "THE BLIND U-ARM BUILD, compared" |
| 2026-10-16 | **E2, the competent-opponent bank, PRE-REGISTERED and running (the study's §16.12)** | §1 (the open list: E2 registered, running); §7 Q2 | quoting any E2 number before its pre-registered read | `design_hidden_info_binding.md` §16.12; `measurements/hidden_info_e2_competent_2026-10-16/PREREGISTRATION.md` (`5881ee83`) |
| 2026-10-17 | **The collapse pilot's honest-vs-oracle STARVATION read (`design_collapse_and_counterfactuals.md` §3.5)** | §3 N16 (an honest one-ply read cannot tell which starved moves are good: precision 1.35× chance, its flagged move −0.068 against the policy's top choice; the oracle's +0.137) | treating honest-world one-ply values as an anti-starvation signal or a completed-Q target today | `measurements/collapse_pilot_2026-10-17/honest_vs_oracle_starvation.{py,json}` (post hoc, zero playouts) |
| 2026-10-18 | **E2 READ (the competent-opponent bank; the study's §16.12)** | Status line; §1 (the never-seen bullet: E2's registered U-FIRST with its scope (SCORER-DEPENDENT, SHARED per checkpoint), the split of the red team's population attack, the moves' necessity share 0.58, the confidence ≈ 65 %; the open list: E2 done, the recommendation to carry R3 into the plateau battery); §5.2 (the U row's E2 tag; the M row's E2 shares and B4); §7 Q2 | quoting U-FIRST without its scorer-dependence and per-checkpoint qualifiers; moving the moves into "what works" (B3 fails on both populations; the moves are necessary, not sufficient) | `design_hidden_info_binding.md` §16.12; `measurements/hidden_info_e2_competent_2026-10-16/` (README §4, `result_e2.json`, `gigo_audit_e2.json`); ledger 2026-10-18 MEASUREMENT (MAJOR) "E2" |
| 2026-10-18 | **The maturity-battery harness BUILT (T35; the study's §12.13.7)** | §7 Q4 (the instrument now exists as tested code: any named run's 25M anchors at its pin, a frozen bank by id and hash, R-MATURITY as rules) | quoting any battery number before its registered read | `src/main/hidden_info_study/maturity_*.py`; `design_hidden_info_binding.md` §12.13.7 |
| 2026-10-18 | **The maturity battery's interim bank and first LEAN trajectory REGISTERED on X26 (the study's §12.13.7)** | Status line; §7 Q4 (the registration, the anchors, the measured cost) | waiting for X26's plateau before any trajectory read; quoting any battery number before its registered read | `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md`, `states_mat_x26.json` |
| 2026-10-18 | **The maturity battery's first LEAN trajectory LAUNCHED on X26** | §7 Q4 (RUNNING, the ETA) | quoting any battery number before `QUEUE_DONE` and the registered read | `measurements/hidden_info_maturity_x26_2026-10-18/README.md` §3 |
| 2026-10-19 | **The maturity battery's first LEAN trajectory READ on X26 (the study's §12.13.7)** | Status line; §1 (a new bullet: maturity does not, so far, bring honest search closer); §7 Q4 (the read, the next step) | quoting the turn ≥ 22 bucket's GAP-SHRINKS (1 of 13 reads at 95 %, no multiplicity correction) or the post-hoc TRUNC-4-K8 TODAY drift as findings; calling the gap's point decline "closer" (it is the oracle's edge falling, not honest search rising) | `measurements/hidden_info_maturity_x26_2026-10-18/README.md` §5; ledger 2026-10-19 MEASUREMENT "the MATURITY BATTERY's first LEAN trajectory" |
| 2026-10-19 | **The WORLD SAMPLER design and its stage-0 screen** (`design_world_sampler.md`, orchestrator brief; DETERMINIZATION added by the owner the same day) | Status line; §1 (a new bullet: the sampler binds, and within it the sets); N15 (the network's species mass at X26 against its moves); §6.1 (illegal worlds MEASURED 5.2 %; the true-dice replay's damage-roll sharpening named; what the exclusions are); Q11 (the opponent-type-per-world matrix and its harness changes); Q12 (the sampler ablation) | quoting stage 0 as a verdict (post hoc, zero playouts, pool opponents); calling the network's species belief "useless" (it is sharp and maturing; the SETS are missing); folding the matrix into the open list ahead of the sampler | `design_world_sampler.md` §5, §6.1, §6.2; `measurements/world_sampler_screen_2026-10-19/` (`result_screen.json`, `result_validate.json`) |
| 2026-10-19 | **The maturity battery's PLATEAU STEP REGISTERED (amendment A1) and LAUNCHED on X26** | Status line; §7 Q4 (the plateau step: P\* declared, the plateau bank, the arms, the CONVERGED amendment, the ETA) | quoting any plateau-step number before `QUEUE_DONE` and the registered read; reading the interim bank's FULL pair as the record (it is the replication) | `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md` §9, README §6 |
| 2026-10-19 | **F2 FIXED: the honest world builder's worlds are legal** (`design_world_sampler.md` §7 F2) | §6.1 (the illegal-worlds bullet: fixed, the after-read, the not-comparable warning; the F1 bullet: the verify step's two tiers) | — | `design_world_sampler.md` §6.1 / §7; `measurements/world_legality_f2_2026-10-19/` |
| 2026-10-19 | **F1 TIER A BUILT, default OFF: dice-marginalised damage rolls in the honest world builder** (`design_world_sampler.md` §3.3.1, §7 F1) | Status line; §6.1 (the true-dice bullet: measured, built, the re-measure) | flipping the default (stage 1's call); tier B now | `design_world_sampler.md` §3.3.1; `measurements/world_rolls_f1a_2026-10-19/` (`result_f1a.json`) |
| 2026-10-19 | **The world sampler's STAGE 1 REGISTERED and LAUNCHED** (`design_world_sampler.md` §6.2) | Status line; §6.1 (the search consumer reads the weights on request; Tier 1's POOL at HEAD is not the pinned pool, F9); §7 Q12 (registered, running) | quoting any stage-1 number before the registered read | `measurements/world_sampler_stage1_2026-10-19/PREREGISTRATION.md` |
| 2026-10-20 | **A LITERATURE REVIEW of search under hidden information** (`../research_state/literature/hidden_info_search_review_2026-10-20.md`; owner P0.1, "what search needs") | Status line; §1 (a literature bullet; the open list REPLACED by the review's ranked L1–L6, with the study's remaining settling items kept); N5 (oracle world only); §5.1 (the calibration qualifier on p\*); §7 Q13–Q15 | appending the review as a second open list beside the red team's (most of that list is done or running); calling p\* "refuted" (the review's reading is judgement, untested until L2 / L4); ranking the Bayesian matrix or any multi-ply information-set method ahead of the anchor and the sampler (the literature puts both after a sampler that is right in distribution) | the review §0, §6, §10, §11 |
| 2026-10-20 | **L1 and L5 READ** (`../research_state/measurements/search_l1_l5_2026-10-20/`; pre-registered `d443c098`) | Status line; §1 (two bullets: the anchored acting rule removes harm but buys no gain; Gen 3 OU near Skat on Long et al.'s map; the "what does not help" list; the open list's L1 / L5 entries replaced by their results); N5 (the honest-world read); §7 Q13 (answered) | adopting an anchored rule as the honest default on the PAYS-SCOPED cell (that cell is the leaf's own E2 signal, not the anchor's); quoting the Tier-1 in-sample fits (+0.005 to +0.013) as gains (they halve or vanish held out); reading L5's placement as a number (proxies on a synthetic-tree map) | `search_l1_l5_2026-10-20/` README §2–§4, `result_l1.json`, `result_l5.json` |
| 2026-10-20 | **L3 READ** (`../research_state/measurements/search_l3_2026-10-20/`; pre-registered `4d149799`; module `main.value_search.opp_likelihood`, `8fe4fb89`) | Status line; §1 (a bullet: the opponent's choices identify the truth; the open list's L3 entry replaced by its result); §7 Q15 (answered) | quoting p_truth 0.249 as what search would get (the truth is never in an honest proposal set: it is the weight the likelihood WOULD give it); reading the snapshot stratum's 0.80 as a ladder number (there the model nearly is the opponent); re-weighting today's 8 honest worlds as stage 1c (S4: it moves ≈ nothing) | `search_l3_2026-10-20/` README §2–§4, `result_l3.json` |
| 2026-10-20 | **STAGE 1c REGISTERED** (`../research_state/measurements/search_1c_2026-10-20/PREREGISTRATION.md`; `design_world_sampler.md` §6.3) | Status line; §1's open list (L3's consequence: registered); §7 Q15 | quoting any 1c number before its registered read | the PREREGISTRATION |
| 2026-10-20 | **L3 CORRECTED after its adversarial review** (`../research_state/measurements/search_l3_2026-10-20/REVIEW.md`, findings F1–F3, F5) | Status line, §1 (the L3 bullet), the open list item 3, §7 Q15: p_truth 0.249 is labelled a COUNTERFACTUAL (the truth inserted among 8 distant honest worlds), not the dose curve's p (the true world's weight among the worlds search plays); the pooled WORKS is stated as carried by the snapshot stratum (bots 0.588, WEAK alone); the read is PROVISIONAL-POOL (one pre-plateau anchor, pool opponents only, familiarity not ruled out) | keeping "0.249 is the range the dose curve asks for" (it compares two quantities: no honest set holds the truth, and p_truth falls with the candidate count); quoting the pooled AUC as a population number (it averages a near-oracle stratum and a WEAK one) | REVIEW.md F1–F3, F5; the familiarity counter-test `search_l3_2026-10-20/counter_test/` |
| 2026-10-20 | **L3 FAMILIARITY COUNTER-TEST READ: NOT-DETECTED, POOL-LIFT** (`../research_state/measurements/search_l3_2026-10-20/counter_test/`, pre-registered `b39c9cdc`) | §1 (two bullets: part of L3 is pool-ness), §7 Q15 | reading NOT-DETECTED as "familiarity ruled out" (decoys beat honest worlds clearly; the truth-vs-decoy interval reaches 0.489); reading it as "L3 is all familiarity" (the point estimate is 0.578 and snapshots show the truth above its decoys) | counter_test README §2, `result_decoy.json` |
| 2026-10-20 | **STAGE 1c PHASE A READ** (`../research_state/measurements/search_1c_2026-10-20/` README §5, `result_1c_a.json`; pre-registered `c9bdc928` + AMENDMENT A `a144482b`) | Status line (the registration's line replaced: it named 64 + 64 proposals, the registered N is 128 + 128); §1's open list (L3's consequence); §7 Q15 | quoting NEARER without its labels (PROVISIONAL-POOL, SELF-PLAY-ONLY: the bot stratum reads nothing); reading the corpus arm's gain as the likelihood's (C8 − F8 alone is the species gain; C128L − C128d is NOT-DETECTED); quoting ΔH as search value (the exact-world mass stays ≈ 0; phase B's playouts are the value read) | README §5 |
| 2026-10-20 | **L3 OFF-POOL REPLICATION READ: REPLICATED-SELF-PLAY-ONLY; off-pool decoy NO-LIFT, TRUTH-SPECIFIC** (`../research_state/measurements/search_offpool_bank_2026-10-20/`, pre-registered `7642df0f6`, bank `5281bfae0`) | Status line (new, latest), the L3 status block, §1 (the L3 bullet: the off-pool sub-bullet), the open list item 3, §7 Q15 | quoting L3's pooled 0.692 or the bots' 0.588 as evidence about a non-self opponent (off-pool the bots read 0.472); reading the pool decoy's 0.575 as real-team coherence (an off-pool real team reads 0.503); treating a pool-bank 1c NEARER as steering (A4: the pool bank cannot separate memory); reading the bot NO-EVIDENCE as covering P\*'s losses (6 of 159 bot states) | the folder's README §2–§6, `result_offpool.json` |
| 2026-10-21 | **The world sampler's STAGE 1 READ: only team identification pays** (`design_world_sampler.md` §6.2) | Status line; §1 (a new bullet); W12 (POOL at 225M); N17 (the open-population samplers); §6.1 (F2's and tier A's IMP effects read: NOT DETECTED); Q12 (READ) | moving any open-population sampler into §2 (NOT DETECTED, not NULL); reading the network-species arm's calibration as value; quoting the exact-world mass as a value proxy (POOL 0.82 exact, TODAY 0.40) | `measurements/world_sampler_stage1_2026-10-19/result_s1.json`, `PREREGISTRATION.md` §9–§11 |
| 2026-10-21 | **STAGE 1c PHASE B READ: NEARER BUT NOT PAYING** (`../research_state/measurements/search_1c_2026-10-20/` README §6, `result_1c_b.json`) | Status line (the latest, read beside stage 1 and L3's off-pool read); §1 (the stage 1c bullet; "what does not help"); the open list; §7 Q15 | quoting phase A's NEARER as a search gain (the playouts read nothing); reading NOT-DETECTED as "no effect" (it excludes only a search-sized ≈ +0.03 gain at n 285); re-running 1c on this bank (A6) | README §6 |
| 2026-10-23 | **The HUMAN OPPONENT MODEL designed and pre-registered** (`design_x50_rung1b_human_opponent_model.md`, X50 rung 1b) | §7 Q15 (the open item "a behaviour-cloned opponent model" replaced by the design, its decisive read R-b2 and its bar) | quoting any HOM number before its registered read; reading R-b1 (held-out humans, completed rows) as the decisive read (only R-b2 runs on true rows); counting the bot stratum as a ladder read (bots switch on 6 % of labels; L3-H, the human-replay read, is unbuilt) | the design §8; R1's README; the off-pool README |
| 2026-10-24 | **R-b2 READ: HOM-NONE** (`../research_state/measurements/x50_r1b_hom_2026-10-23/`, registered `42f1d9a7` §8.4; G-REPRO 320 / 320 bit-exact) | Status line (new, latest); §1 (the off-pool L3 bullet: the behaviour-cloned candidate is now tested); §7 Q15 | reading HOM-NONE as "humans' choices carry no evidence" (the bank's opponents are scripted bots: F-1b-3; L3-H is the ladder's read, unbuilt); reading R-b1's −0.58 nats as a search gain (a gate on completed rows); un-pausing search | the README §R-b2, `rb2_read.json` |
| 2026-10-25 | **The next-step DESIGN written (DRAFT, `design_search_next.md`; no measurement)** | the pointer line under the banner; no row, verdict or number here changes | adding its derived bars (the bridge, the identification target) to §4 / §5 before E2 measures the bridge (they are UNVERIFIED derivations) | `design_search_next.md` §1.2, §1.6, §3 |
| 2026-10-25 | **The maturity battery's PLATEAU STEP READ: GAP-SHRINKS NOT DETECTED on the record** (`../research_state/measurements/hidden_info_maturity_x26_2026-10-18/` README §7, `result_mat_plateau.json`; amendment A1; GIGO 531 / 531) | Status line (new, latest); the 2026-10-19 plateau status (RUNNING → READ); §1 (the maturity bullet); §5.1 (TODAY at the plateau); §7 Q4; rule 3 (the battery has run; per-claim relabel not done); L1's consequence (now unblocked) | quoting the interim pair's GAP-SHRINKS as the verdict (it is the replication; the record is NOT DETECTED); reading STILL-MOVING NOT DETECTED as convergence; quoting any cell as a real-opponent size (all SELF-REFEREE) | README §7; `gigo_audit_mat_plateau.json`; ledger 2026-10-25 MEASUREMENT |
| 2026-10-25 | **E1-0 READ: UNDECIDED** (`../research_state/measurements/search_e1_0_2026-10-26/`; registered before any bot playout; every VOID guard passed) | Status line (new, latest); §1 POOL bullet and W12 (labelled SELF-REFEREE, with the real-bot number beside); §6.2 (the referee's self-play and its new `opp_bot` option) | relabelling the bot-stratum IMPs as misread (no primary cell disagrees: UNDECIDED, so §9 of the registration says no relabel); reading the 23 disagreeing SECONDARY cells or G_F's D as a verdict (95 %, uncorrected over 108 cells); re-scoring the plateau step's rows (their registered read has not been made) | `search_e1_0_2026-10-26/README.md`, `result_e10.json`; `design_search_next.md` D-1, E1; its review F8 |
| 2026-10-25 | **L1's REPLICATION on the plateau bank READ: the PAYS-SCOPED cell REPLICATES (thin), SELF-REFEREE** (`../research_state/measurements/search_l1_plateau_2026-10-25/`; the cell, bar and frozen Tier-1 fit written before the read, `1f980728a`; GIGO 0 / 1,438 mismatches) | Status line (new, latest); L1's status paragraph; §1 (the L1 bullet); the open list item 1; §7 Q13 | adopting an anchored rule on the replicated cell (the anchor's own part is +0.0025 [−0.0039, +0.0089], not clear); quoting +0.0097 as P\*'s size (it is the 25M anchor's; P\* reads +0.0001); quoting any cell as a real-opponent size (SELF-REFEREE) | README §3, FINDINGS 1–4; `result_l1_plateau.json`; ledger 2026-10-25 MEASUREMENT |
| 2026-10-27 | **H-MID entered; S-U PROPOSED** (search-research design worker, orchestrator brief; docs only) | Status line (new, latest: H-MID's CRITIC-MISCALIBRATED-ON-U with V's intact ordering and its contested search reading); the pointer to `design_search_u_headroom.md` under the next-step design's; §7 Q17 | reading H-MID as "search on V cannot help" (the forensics' lever map) or as "search on V has headroom" (the brief's): V's game-level AUC does not measure its within-node action ranking; quoting S-U's costs as measured (they are estimates from other reads' per-decision costs) | `../research_state/measurements/h_mid_value_test_2026-10-26/README.md` (`f200258f`) FINDING 1, reading 2–3; `../research_state/measurements/search_offpool_bank_2026-10-20/README.md` (G-OFF: the traced side on a pool team); `design_search_u_headroom.md` |
| 2026-10-27 | **S-U's live searched player BUILT (B1–B3), smoke-tested** (search engineer, orchestrator brief; no read played) | §6.4 (new: the instrument, its honest-log cut and guards, G-PREFIX, the peer's patience, the smoke's cost); the S-U pointer line; §7 Q17's cost cell. piKL is now `SearchConfig(rule="pikl")` and re-scores L1's banked rows exactly | quoting the smoke's 6.4 CPU-s a decision as S-U's cost (12 decisions, 2 deals: stage 0 measures it); deciding on the event loop (the negative control dropped the peer and booked a forfeit win) | `src/main/anchors/search_side.py`, `src/main/reads/prefix.py`, `src/main/value_search/rules_pikl_test.py`; `~/gen3ai_archive/su_build_smoke_2026-10-27/`; `design_search_u_headroom.md` §6, §9 |
| 2026-10-27 | **Q17: "H-MID motivates S-U" softened** (doc editor, orchestrator brief; docs only) | §7 Q17: H-MID does not rule S-U out and does not motivate it (the search leaf is the self-play regime, where the premium cancels); S-U is an independent question | "H-MID motivates it" | `../research_state/measurements/h_mid_value_test_2026-10-26/REVIEW.md` M4 |
| 2026-10-27 | **V-RANK PROPOSED, S-U's filter** (research designer, orchestrator brief; docs and a < 1-CPU-min probe on banked rows) | The S-U pointer paragraph (its filter); §7 Q17's question and cost cells | a self-referee ranking read on U (H-MID REVIEW M1: it grades Q̂ against the same blind spot); a gain-facing decision-level primary (≈ 10⁵ positions a set for a game bar) | `design_v_rank_on_u.md`; `../research_state/measurements/v_rank_on_u_2026-10-27/power_probe.json` |
| 2026-10-27 | **The live search's cost MEASURED and cut, bit-identically: the LEAN world screen** (search-performance engineer, orchestrator brief) | §6.1 (the screen's cost and the lean screen), §6.4 (Cost), the S-U pointer line and §7 Q17's cost cell (≈ 560 CPU-h, was ≈ 1,000; the old builder's ≈ 2,000–3,000). No verdict moves: the worlds, draws and picks are unchanged | replaying the true log's prefix once (a world's battle leaves the truth's at the screened mon's first switch-in); dropping `core_events`' cross-checks from screens (not identical by construction) | `../research_state/measurements/su_headroom_2026-10-27/AMENDMENT_1_DRAFT.md` §1b; `src/main/value_search/world_screen_parity_test.py` |
| 2026-10-27 | **The lean world screen REVIEWED adversarially: no BLOCKER** (adversarial reviewer, orchestrator brief; docs, two slow tests, CPU probes) | §6.1: "bit-identical" is conditional on no core refusal after a screen's stop (measured 0 / 14,746, now gated by `world_screen_parity_test`'s per-replay invariant and wrong-world stress) | scoring a refused replay by its shipped lines (identity by construction; a reference-semantics change, left to the search owner) | [`reviews/lean_world_screen_review_2026-10-27.md`](reviews/lean_world_screen_review_2026-10-27.md) |
| 2026-10-27 | **The S-U gate test BUILT** (S-U gate operator, orchestrator brief) | §6.4's offline-at-kept-states line: raw cells keep honest input logs; S re-run offline exactly as the live searcher; deviations scored by the self-referee; rule `deviation_sign` | a zero-playout gate (none exists for U states) | `../research_state/measurements/su_headroom_2026-10-27/REVIEW.md` § The gate test |
