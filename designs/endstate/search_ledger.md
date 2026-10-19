# THE SEARCH LEDGER: what we know about search, where it works, where it does not, and how far we are

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from
this doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-19, latest): the maturity battery's PLATEAU STEP is REGISTERED and RUNNING** (`design_hidden_info_binding.md` §12.13.7; `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md` §9, README §6). P\* = 280M is declared. A PLATEAU bank of 800 states is drawn from P\*'s cycles. P\* runs FULL on both banks, with the hide-only O−M / O−U arms, so R3's plateau replication has both halves; 25M runs FULL on both banks. The plateau bank's [25M, P\*] read is the verdict of record. CONVERGED is read only pooled (not in this step), and STILL-MOVING replaces it. The queue was LAUNCHED 2026-10-19 05:20 PDT; ETA `QUEUE_DONE` ≈ 2026-10-23 to 24 (README §6.3). No number yet (§7 Q4). **Status (2026-10-19): the MATURITY BATTERY's first LEAN trajectory has been READ** (`design_hidden_info_binding.md` §12.13.7; `measurements/hidden_info_maturity_x26_2026-10-18/README.md` §5; PROVISIONAL, interim bank). On the same 287 states at X26's 25M, 100M, 175M and 225M checkpoints, the oracle − honest gap G_F falls +0.075 → +0.062 → +0.055 → +0.053, but GAP-SHRINKS is NOT DETECTED (−0.022 [−0.049, +0.004]). The network's belief does not rise (TODAY ≈ 0.01 at every anchor; BELIEF-RISES NOT DETECTED), the series is NOT CONVERGED, and the FRONTIER is NOT READ in LEAN (every bucket's gate CLOSED at every anchor). Post hoc: the decline is the ORACLE's edge over the policy falling, not honest search rising (honest search stays at or below the policy); it is flat at the 4-turn leaf. Maturity alone is not bringing search closer. §1, §5.1 and §7 Q4 carry it. **Status (2026-10-18): the MATURITY BATTERY is REGISTERED on X26** (`design_hidden_info_binding.md` §12.13.7; `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md`). An INTERIM frozen bank of 800 states (LEAN 320 nested; half from games against our own snapshots, as E2 recommends) is drawn from X26's newest cycles, because X26 has no plateau yet. The plateau bank, when it exists, supplements it and becomes the verdict of record. LEAN reads at 25M, 100M, 175M and 225M ask whether honest search's gap to the oracle shrinks and whether the network's belief rises as the policy matures. No number yet (§7 Q4). **Status (2026-10-18): E2, the COMPETENT-OPPONENT bank, has been READ** (`design_hidden_info_binding.md` §16.12;
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
- **Maturity does not, so far, bring honest search closer** (the maturity battery's first LEAN trajectory on X26, `design_hidden_info_binding.md` §12.13.7; PROVISIONAL, interim bank, read 2026-10-19). On 287 frozen states at 25M / 100M / 175M / 225M:
  - G_F (oracle − honest, LEAF-K8) +0.075 / +0.062 / +0.055 / +0.053; the change −0.022 [−0.049, +0.004] is NOT DETECTED.
  - Honest search's own gain over the policy stays ≤ 0 (IMP_F −0.012 / −0.022 / −0.004 / −0.011).
  - The point decline is the oracle's edge over the policy falling (ΔIMP_O −0.021 [−0.046, +0.006]), not honest search rising (ΔIMP_F +0.001); at the 4-turn leaf the gap is flat (post hoc).
  - The network's belief is worth p ≈ 0.01 on the true world at every anchor, against p\* ≈ 0.2–0.3 (BELIEF-RISES NOT DETECTED).
  - Every stage's gate is CLOSED at every anchor; the frontier is NOT READ in LEAN.
  - What would enable search is the same at 225M as at 25M: a world sampler that puts ≈ 0.2–0.3 of weight on the true world.
- **Information is a SEARCH lever, not a POLICY lever, at this maturity** (the knock-in forks, `design_hidden_info_binding.md`
  §8.7; PROVISIONAL: +3M steps from one 15M parent, no placebo). A network TRAINED 3M more steps with the opponent's true
  moves, items or never-seen species in its observation is no stronger than its control: moves −0.38 pp [−1.86, +1.10], items
  +0.13 [−1.61, +1.86], species −0.59 [−1.77, +0.58] (NULL); spreads +0.34 [−2.51, +3.18] (NOT DETECTED). Withheld at play,
  the moves and species forks are ≈ 2.4–2.6 pp WORSE than the control (descriptive): they learned to rely on the told facts,
  not to gain from them. So belief work pays, if anywhere, as a world sampler for search (G +0.070), and privileged
  information, if ever used in training, belongs in the critic only.
- **What does not help:** more rollouts as critic-training labels, a per-action head on noisy oracle labels (−0.004), the
  proof-first 2·SE acting rule, trusting the policy prior at Gumbel's default scale, shared-seed dice pairing beyond one
  turn, a privileged critic, telling the frozen network the true moves (+0.001), training it 3M steps with them (−0.38 pp), the network's belief as a world sampler,
  the honest sharpening (prior-mode moves), the pool's per-species sets, and Foul Play's hand evaluator as the leaf in
  honest worlds (no better than the critic).
- **Open, in order of cost.** The red team (§16.10) has run, and so has its E1 (the blind U-arm build, compared in §16.11:
  AGREE, scoped). Its settling list, in order:
  - the second referee (≈ 3–5 CPU-h);
  - ~~a competent-opponent state bank with R3 pre-registered on it (E2)~~ **READ 2026-10-18 (§16.12): U-FIRST, scorer-dependent,
    PROVISIONAL** (above). The reader recommends no more CPU on U vs M at 15M, and carrying R3 into the plateau battery with
    competent-opponent states in its frozen bank (§7 Q4);
  - species-only never-seen worlds (F+Usp, ≈ 15 CPU-h);
  - Tier 2 (items / abilities / spreads arms);
  - the maturity battery at plateau (§7).

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
| W12 | **Identifying the opponent's TEAM recovers most of the gap, on our closed pool.** Worlds drawn from the training-pool teams consistent with the observation (POOL), gated | MEASURED (Tier 1, R-RECOVER BELIEF-RECOVERABLE on this distribution) | r_POOL **0.80** [0.73, 0.87]: POOL IMP +0.063 vs oracle +0.076; the pool's per-species SETS alone (POOLSETS) recover −0.02: what pays is knowing WHICH TEAM, not the set priors. Not available on the ladder (an open population) | `design_hidden_info_binding.md` §16.8 |

---

## 3. WHAT DOESN'T WORK, and why

| # | what | tag | evidence | why (a mechanism, and whether it is measured) |
|---|---|---|---|---|
| N1 | **The old proof-first ACTING RULE** (switch only when the paired difference is resolved at 2·SE) | MEASURED (PRE-ERA, reproduced on rb-blob and in the pilot) | resolved 4.5 % / 12.5 % of pairs at R = 4 / 16; changed 1 of 250 live decisions; rb-blob 6.7 / 15.3 / 17.7 / 26.9 % at R = 4 / 8 / 16 / 32. GATE-2SE at K8 keeps +0.050 / +0.041 of ARGMAX's +0.081 / +0.071. Opening it to 0.5·SE bought 6.2× more offline, but the live read (0.4719 [0.402, 0.542]) is VOID (a recon-tag collision wrote fabricated choices into the live record) | the question "is this pair resolved" is the wrong one: decisions need the best of ~10 siblings, not a certified pair. MEASURED (ledger 2026-09-19 L19885, 2026-09-20 L19994, 2026-10-11 L24263) |
| N2 | **More rollouts as critic-training labels** (a frozen readout refit; richer-label arms; λ-returns; contested-state forks into the buffer) | MEASURED (CRN refit on rb-blob; the rest PRE-ERA) | REFIT-K8 − BASE **−0.014** [−0.032, +0.002]; K8 vs K1 label no difference; arm 10 R-rollout targets NULL at 10M; λ-return 0.9 not confirmed; fork arm Δ −0.0003 [−0.0148, +0.0136]; twelve heads on 0.567–0.578 | the readout is not label-limited (a head fitted in 1.1 s matches one trained 75M steps); the label buys calibration (ECE 0.065 → 0.032), not resolution. Whether a TRAINED trunk on sibling-level targets differs is UNVERIFIED (ledger 2026-10-06 L22512, 2026-09-19 L19663) |
| N3 | **A per-action head on noisy oracle labels** (frozen end-state body, 2 oracle playouts per action, 9,629 states) | PROVISIONAL, end-state only | R-QHEAD **−0.004** [−0.016, +0.009] NOT DETECTED; R-QHEAD-CRITIC **−0.018** [−0.032, −0.004] BEHIND (v2 builder, confirmed); validation R² +0.002 | **83 %** of the labels' sibling spread is dice noise (R² ceiling 0.17) and most of the rest is hidden information; a head that sees only our observations can learn ~0.5 % of the variance. How many labels would suffice is UNVERIFIED; a trained body is untested (ledger 2026-10-11 L24485) |
| N4 | **Shared-seed (CRN) pairing across a full playout** | MEASURED | VRF **1.057** [1.045, 1.072] at the terminal (move-move pairs 1.28, move-switch 1.04); ρ after one turn 0.44–0.53, at the terminal 0.06; the derangement VRF 1.000 [0.996, 1.005] so the unpaired variance is estimable from shared-seed data | ONE PRNG per battle: shared seeds share the STREAM, not the roll-to-event MAPPING, and the first differing draw misaligns every later one. R-PAIR reads ALIGNMENT-LOSS ⇒ the registered rule says build S1 (per-turn reseed); NOT built. The AIVAT-style branch-turn control variate buys nothing (CV-K8 − LEAF-K8 ≈ 0) (`design_value_discrimination.md` §5) |
| N5 | **Gumbel's prior-aware selection at the `mctx` defaults (c_visit 50, c_scale 0.1), and top-k restrictions / sequential halving** | PROVISIONAL | R-PRIOR **−0.013** [−0.021, −0.005] / **−0.022** [−0.031, −0.013] (K4); at K8 −0.022 / −0.031; SH-4 is NOT DETECTED at a small budget and WORSE (−0.011…−0.027) at larger ones. A post-hoc read (`target_scale_read.json`): at c_scale ≥ 1 sampling π′ recovers the full ARGMAX gain | where the leaf disagrees with the policy, the leaf is right; at 0.1 our win-scale Q (a spread of ~0.1–0.2) is too small against the policy's logits, so the rule under-trusts it. The module now defaults to c_scale 1.0 (the calibration reading is POST HOC) |
| N6 | **Depth-1 search over the critic of the time, in fair worlds (PRE-ERA)** | MEASURED, PRE-ERA | depth-1 search COST 17–22 points of mirror win rate (honest@1 s 0.294 [0.24, 0.35]); flat across a 32× dice sweep (0.125 … 0.205): BIAS, not variance; 13× more certified overrules moved the win rate onto the null (0.5003 [0.4803, 0.5203]); no win-prob head paid as a leaf at 3 s (0.506 / 0.503 / 0.495) | search AMPLIFIES the critic's systematic error; the leaf, not the allocator, was the limit. The current-era re-measure is §2 W5–W6 (ledger 2026-08-24 L04104, L04318, 2026-08-29 L05912, 2026-09-11) |
| N7 | **The asymmetric (privileged) critic** | MEASURED, PRE-ERA, one run, no replicate | `truevalue` (V given the opponent's TRUE team) at 10M: NOT DETECTED on discrimination (turn-1–3 spread ratio 0.00 vs 0.09–0.12); +0.040 (all) / +0.088 (late) more OPTIMISTIC than its own continuation; ECE +0.04 | hypothesis only: a critic conditioned on information the policy never has is confidently wrong about what the policy will do. The information family is CLOSED for the era; the omniscient twin (X11) stays offline, never a training target (`UNDERSTANDING.md` §4.x; `design_q_head.md` §7) |
| N8 | **Telling the NETWORK the true moves** (frozen; and trained) | PROVISIONAL (frozen; the knock-in forks); MEASURED (X5 oracle arms, below the floor) | injected at 95 %: it flips 13 % of picks and gains **+0.001** [−0.003, +0.005] (R-NET, CRITIC-USES NOT DETECTED, 3 networks); a wrong-moves placebo costs the critic search ≈ 0.8 points. X5 oracle arms: told species +2.44 pp [+0.31, +4.57], full set +2.18 pp [+0.53, +3.84], the set over species −0.26 [−1.92, +1.40], all below the 4.57 pp replicate floor; an untrained network handed a revealed team FELL −8.1 pp. **Knock-in forks** (3M more steps of the end state with the truth at 95 % in its observation, 3 seeds per arm vs a no-change control, `design_hidden_info_binding.md` §8.7): moves **−0.38 pp** [−1.86, +1.10], items +0.13 [−1.61, +1.86], never-seen species −0.59 [−1.77, +0.58] (NULL), spreads +0.34 [−2.51, +3.18] (NOT DETECTED); MOVES-FIRST false; no placebo. Withheld at play: moves −2.35, species −2.58 pp vs the control (told − withheld ≈ +1.9, descriptive) | a USABILITY limit, not an information one. The frozen network does not convert the move content. A trained one learns to READ it (removing it costs ≈ 1.9 pp) but not to GAIN from it: it substitutes the told facts for its own inference (mechanism UNVERIFIED: distribution shift, learned reliance or capacity; §8.7 lists the separating experiments) |
| N9 | **Smogon-usage or network-belief worlds as a sufficient sampler** | PROVISIONAL | R-HONEST NOT DETECTED (+0.011 / +0.007); R-BELIEF (network species belief − Smogon) −0.010 [−0.024, +0.005] / −0.002; the network's MOVE belief closes ~⅕ (end-state) / ~1⁄20 (production) of the oracle gap (R-MOVES +0.017 [+0.001, +0.031] HELPS end-state only, NOT DETECTED on production, builder-sensitive) | the belief is barely sharper than Smogon usage (51 % / 48 % of a world's hidden moves right vs 45 %); a better species hit rate (0.45 vs 0.21) buys NO value (B′ ≈ 0) because value needs the whole world to hang together |
| N10 | **"Focus the search on torn states"** | PROVISIONAL | LEAF-K8 flat across the policy's gap quintiles (q1 +0.078 / +0.090, q5 +0.075 / +0.072); banked: torn 40 % +0.049 vs most decided 20 % +0.082. No cheap policy-confidence feature separates the decisions search flips (PRE-ERA: flip rate 0.694 [0.681, 0.707]) | the stakes are larger where the policy is confident |
| N11 | **Narrowing the state distribution to buy discrimination** (an exploiter's head) | MEASURED, PRE-ERA | sibling ranking 0.575 on its own slice and the generalist's alike | the 0.575 ceiling was four heads deep: coverage, width, loss form and data each varied |
| N12 | **Hand evaluators as the leaf (PRE-ERA)** | MEASURED, PRE-ERA | nine tried; none beat the head; the PBRS potential matches a 75M critic pooled | Foul Play's edge is evaluator, search shape or knowledge: not separated (§7) |
| N13 | **Foul Play's hand evaluator as the leaf, current era, honest worlds** (R-EVAL) | MEASURED (Tier 1) | in F: HE − critic +0.007 [−0.001, +0.016] (4-turn leaf), −0.002 [−0.011, +0.006] (one ply); in the pilot's P: +0.003 / −0.005: **not EVALUATOR-BOUND**; the hand eval also collapses oracle → honest (g_HE / g_CRIT 0.75 [0.52, 1.06]): INFORMATION-BOUND support. **RED TEAM (§16.10): WEAKENED**: the 4-turn contrast is a near miss TOWARD evaluator-bound (one-sided tail 5.5 % at the +0.0005 bar; positive on both checkpoints); not replicated | information binds; a hand leaf ≈ 0.7 points better than our critic in honest worlds is NOT excluded |
| N14 | **The seen mons' hidden MOVES as THE binding constraint** (R-MOVES) | MEASURED (Tier 1): **REFUTED (R3)** | s_M 0.13 [−0.02, 0.25], n_M 0.42 [0.31, 0.54] vs s_U 0.31, n_U 0.82 (U dominates on both sides, paired: +0.19 [+0.03, +0.35], +0.41 [+0.26, +0.55]); the moves' dose helps a little (D90 − F +0.011), a wrong set hurts (W90 − F −0.013). **On E2's fresh competent-opponent states too** (§16.12): R3 U-FIRST (scorer-dependent), s_M 0.16 (B3 fails), n_M 0.58 (B4 holds: necessary, not sufficient) | the never-seen Pokémon carry more; a move-inference programme is not the lever (`design_hidden_info_binding.md` §16.8) |
| N15 | **The network's own belief as the world sampler on the honest base** (FN) | MEASURED (Tier 1) | FN LEAF-K8 +0.006 = F's +0.006 (4-turn +0.013 vs +0.005); its move hit 0.513 vs Smogon 0.508, species (a never-seen slot holding a TRUE never-seen species) **0.411 vs 0.180** (the published 0.172 vs 0.058 counted a species against the truth of the slot it was DRAWN for: the 2026-10-15 builder-record GIGO, `design_information_levers.md` §7.6; the red team's count from the world teams agrees). FN − F (red team, §16.10 of the study): +0.003 [−0.006, +0.012] at LEAF-K8, +0.008 [+0.0001, +0.017] at TRUNC-4-K8 (a BOUNDARY result) | barely sharper than Smogon on MOVES; on SPECIES 2.3× sharper, and that buys at most ≈ +0.017 at any scorer (the whole world must hang together) |

| N16 | **An HONEST-world one-ply read as a STARVATION detector / anti-starvation target** (which starved near-best moves are good?) | DESCRIPTIVE, POST HOC (on Tier 1's arm F and the pilot's truth; `design_collapse_and_counterfactuals.md` §3.5) | on decisive turns, flagged starved moves are truly near-best at precision 0.260 (F, 8 worlds) vs 0.194 chance and 0.501 for the ORACLE 8-rollout leaf; playing F's flagged move instead of the policy's top choice **−0.068** [−0.105, −0.031] (oracle +0.137 [+0.104, +0.169]); the critic one-ply on F worlds −0.035 | the value of the moves the policy starves is hidden information (the never-seen mons first, N14); at 8 samples "within 0.1 of the best" is "tied with the best", and F's per-action SE is 0.24. A counterfactual TARGET from honest one-ply values would feed the wrong moves |
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
- **Along X26's maturity (the battery's first LEAN trajectory, 2026-10-19; PROVISIONAL, interim bank; §1).** The same curve on X26's 25M / 100M / 175M / 225M checkpoints, 286 frozen states:
  - TODAY (registered, LEAF-K8) is 0.00 / 0.01 / 0.00 / 0.01: below the tie-break step at every anchor (BELIEF-RISES NOT DETECTED, +0.011 [−0.034, +0.178]).
  - P-STAR against the policy (post hoc, R-DOSE's definition) is 0.25 / 0.30 / 0.20 / 0.30.
  - On the continuous 4-turn leaf (post hoc, not registered) TODAY drifts 0.02 → 0.18 [0, 0.36], while P-STAR drifts out 0.35 → 0.50 as the oracle's ceiling falls.
  - The distance to p\* does not close with training.
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
(39 built, 1 excluded); cost ≈ 28 s per decision at 4 worlds on a contended box (the honest screen dominates). Offline / evaluation only;
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
were not plateaued.

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

---

## 7. OPEN QUESTIONS, and the experiment that settles each

| # | question | the experiment | cost / status |
|---|---|---|---|
| Q1 | Does a FULLY honest search beat the critic anywhere, and which class of fact carries the gap? | **Tier 1 of the hidden-info study** — READ 2026-10-15 (§16.8): no (FRONTIER EMPTY; k = 6 GATE-WEAK); the never-seen Pokémon (R-MOVES REFUTED by R3). **Tier 2** (F+I/A/S, O−I/A/S, D25 / D75, W50 / W100, MAP90, P16 = R-WIDTH, the second referee) is the open remainder | Tier 1 cost ≈ 1.5 days on 3 agent cores (FN built; the honest screen ≈ 20–60 CPU-s per state). Tier 2 not queued |
| Q2 | Is the pilot's oracle − honest gap an artifact of our instrument? | **The blind replica's third-party comparison** (R-REPLICATE, then the red team, `design_hidden_info_binding.md` §13) | **PARTLY ANSWERED 2026-10-15 (§16.9):** the gap is NOT an artifact of one builder (G replicates on the same states). R-REPLICATE as registered is NOT MET: the ν_M point rule fails, and TV < 0.02 is below the noise floor. R3's never-seen half is unreplicated. **The RED TEAM has run (§16.10):** the instrument holds against every attack it could make offline (tie-free scoring, the exclusions, the mis-keyed records, a U-leak check, the U arms' team text). What it cannot settle offline is the never-seen half: the blind U-arm build E1 (16–40 CPU-h, ready to dispatch in `red_team.md` §3), the competent-opponent bank E2 (40–65 CPU-h) and the second referee E3 (3–5 CPU-h). **E1 READ 2026-10-16 (§16.11):** AGREE by its pre-registered criterion at LEAF-K8 on the 978 common states, so R3's never-seen half is REPLICATED, scoped to the 8-rollout leaf and the bot-heavy registered mix (both builds fail it on snapshot-game states and on end2 alone). R-REPLICATE §12.2 as a whole is still NOT MET on the M side. **E2 READ 2026-10-18 (§16.12):** U-FIRST on 1,200 fresh competent-opponent states by the registered pooled LEAF-K8 primary, SCORER-DEPENDENT and SHARED per checkpoint; the GIGO audit is clean. Open: E3 (the second referee), and the plateau replication (Q4) |
| Q3 | Can a network LEARN to use told information, and what is each class worth in strength? | **The knock-in forks**: `--oracle-reveal ki_moves` / `ki_items` / `ki_spreads` / `species:unseen@0.95` with `--diagnostic-fork-consent`, 5 arms × 3 seeds × +3M from the adopted end state, a no-change control (the v8 guard); no placebo (the observation refuses one) | **ANSWERED at +3M, PROVISIONAL (2026-10-15, `design_hidden_info_binding.md` §8.7):** it learns to READ moves and species (withheld costs ≈ 1.9 pp) but GAINS nothing in strength (moves, items, species NULL; spreads NOT DETECTED). Open: whether longer exposure, a critic-only route or an annealed reveal leaves a durable honest gain; which mechanism carries the dependence (§8.7's experiments (a)–(e), none queued) |
| Q4 | What do maturity and plateau change: does G shrink, does today's p rise toward p\*, does the frontier move earlier? | **The maturity battery** on the X26 deep run (`design_hidden_info_binding.md` §12.13.7): O, F, FN, F+M, F+U re-run at ≥ 25M then every 25M and at the plateau checkpoint, ONE frozen 800-state bank, rules R-MATURITY (GAP-SHRINKS / BELIEF-RISES / FRONTIER-MOVES-EARLIER / -STATIC / -RECEDES). **Harness BUILT 2026-10-18 (T35)**; **the first trajectory REGISTERED 2026-10-18**: an INTERIM 800-state bank (LEAN 320; half own-snapshot states) from X26's 230M–234M cycles, LEAN at 25M / 100M / 175M / 225M, the plateau bank to supplement it (`measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md`). Measured LEAN ≈ 7–8 CPU-h per anchor (not 14); **the first LEAN trajectory READ 2026-10-19** (README §5; PROVISIONAL, interim bank): GAP-SHRINKS NOT DETECTED (G_F +0.075 → +0.053, change −0.022 [−0.049, +0.004]), BELIEF-RISES NOT DETECTED (TODAY ≈ 0.01), NOT CONVERGED (unreachable at the registered sizes: a FINDING), FRONTIER NOT READ (LEAN), every gate CLOSED. **The PLATEAU STEP REGISTERED and RUNNING 2026-10-19** (amendment A1, PREREGISTRATION §9; README §6). It runs P\* = 280M FULL + O−M / O−U on the interim and the new PLATEAU bank (800 states from P\*'s cycles), and 25M FULL on both. The record is the plateau bank's [25M, P\*]. CONVERGED is read only pooled over both banks, so it is not read in this step; STILL-MOVING replaces it. ≈ 150 CPU-h, ETA ≈ 4–5 days. The code: `src/main/hidden_info_study/maturity_{series,bank,rules}.py` (the 25M trigger, the read-only checkpoint copy, the frozen bank sampler, R-MATURITY as code), drivers and queue in `measurements/hidden_info_maturity_x26_2026-10-18/` | ≈ 150 CPU-h for six checkpoints (FULL ≈ 47 at the first and plateau anchors, LEAN ≈ 14 between), no GPU. Every claim in this ledger is PROVISIONAL until it runs |
| Q5 | Does the honest ONE-PLY critic search pay in GAMES? | **S0b** (`design_value_discrimination.md` §13.4): the end-state checkpoint with an 8-world one-ply critic search vs the same checkpoint raw, mirrored head-to-head; pass = lower bound > the +4.57 pp replicate floor | ≈ 1 CPU-day. NOT run as far as the ledger records; OWNER approval requested (§13.0 ask 2) |
| Q6 | Is the stage-gated bootstrap viable: does honest search work anywhere well enough to label? | **R-STAGE-SEARCH** (Tier 1S, 13 buckets, per checkpoint and pooled at 99.6 %), then the S1 SHADOW (logs only, per-stage `search/imp`), then the pre-registered A/B (`design_value_discrimination.md` §13.4, §13.7, §13.10) | A DESIGN PROPOSAL for the owner's review, nothing built. Gate closed today (§4). The in-training deferral (2026-09-30) is NOT lifted by this doc |
| Q7 | The oracle-critic roles: may an oracle-informed critic or oracle labels help? | (a) the offline omniscient twin X11 as a measuring instrument (V − E_belief[V\*]; where is error reducible, where is information worth gathering), never a target; (b) oracle playouts as value labels for a TRAINED body or a much larger label budget (`design_value_discrimination.md` §13.8, GPU; needs the owner's ruling on in-training search). The 09-10 `truevalue` arm (N7) argues against training the public V on true values | not run; (b) is the only route that reaches the hidden-information-limited label |
| Q8 | Is the move-belief gap closable at all? | ANSWERED for search at 15M (Tier 1, 2026-10-15): the moves are a minority share (N14) and the network's belief worlds buy nothing (N15); today's belief sits below the tie-break step (§5.1) | for the NETWORK: the knock-in forks (Q3) read NULL at +3M. Even the told truth does not make the policy stronger, so closing the move-belief gap is not a policy lever at this maturity |
| Q9 | Is the evaluator or the information the limit (Foul Play's hand evaluator vs our critic)? | **R-EVAL** — READ 2026-10-15: not EVALUATOR-BOUND, INFORMATION-BOUND support (N13); width (Foul Play's other lever, R-WIDTH / P16) is Tier 2 | width open |
| Q10 | Does the full-length leaf need the per-turn reseed (S1)? | R-PAIR said ALIGNMENT-LOSS; S1 (~0.5–1 agent-day) matters only if a full leaf is still wanted after the 4-turn leaf | not built; conditional |
| Q11 | What does one-ply best-response give up against an equilibrium method? | the cheapest equilibrium-flavoured step: an MMD-style magnet term (`design_value_discrimination.md` §14.3), or the per-turn matrix-game solve (§14.4) | not built, not registered |

---

## 8. THE MAINTENANCE RULE

1. **Every search-related result updates this ledger in the same commit**: a new IMP, a stage cell, a gate verdict, a builder
   change that moves a leak, a replica or red-team finding, a fork read, a maturity checkpoint. Add or change the row, move
   the claim between §2 and §3 if its verdict flips, and say what changed in the Decision record.
2. **Claims cite ledger entries** (`research_state/ledger.md`, found through `ledger_index.md`) or the design section that owns
   the number; this doc never carries a number its owner does not. The ledger wins any disagreement; fix the view, never the ledger.
3. **Pre-plateau results are labelled PROVISIONAL**; they stay so until the maturity battery (Q4) has run at the plateau
   checkpoint. A pre-Rustboro result is labelled PRE-ERA, kept as history, and paired with its current-era re-measure.
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
| 2026-10-19 | **The maturity battery's PLATEAU STEP REGISTERED (amendment A1) and LAUNCHED on X26** | Status line; §7 Q4 (the plateau step: P\* declared, the plateau bank, the arms, the CONVERGED amendment, the ETA) | quoting any plateau-step number before `QUEUE_DONE` and the registered read; reading the interim bank's FULL pair as the record (it is the replication) | `measurements/hidden_info_maturity_x26_2026-10-18/PREREGISTRATION.md` §9, README §6 |
