# THE SEARCH LEDGER: what we know about search, where it works, where it does not, and how far we are

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from
this doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-12): ESTABLISHED at the owner's request** ("try and build a ledger of what we know about search, what
we know about where it's working, where it's not working, how much we need to improve for it to work … establish the
ledger now"). This is an INDEX-AND-VERDICT doc: it holds no measurement of its own. Every row points at the ledger
entry, design section or measurement folder that owns the number, and the owner wins any disagreement
(`research_state/ledger.md` wins over everything; `design_hidden_info_binding.md` and `design_value_discrimination.md`
own the pilot's detail). Not read in writing it: the BLIND REPLICA's code and results
(`src/main/hidden_info_replica/`, `measurements/hidden_info_replica_2026-10-11/`); its comparison is an open item (§7).

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
- **Fair (honest) search does not yet beat the critic.** With worlds guessed from Smogon usage and kept only if they
  replay everything our side saw, an 8-rollout leaf gains +0.011 / +0.007 (not clear of 0 or of the critic); the whole
  oracle edge is hidden information about the opponent's TEAM (oracle − honest +0.070 / +0.065). PROVISIONAL, and the
  honest numbers are OPTIMISTIC: the pilot's worlds kept every seen mon's item, ability and spread TRUE (§5, §6).
- **It is a frequency effect, concentrated early and with material left.** The value of perfect information G falls
  +0.086 → +0.035 from 1–2 to 6 opponent Pokémon seen because the truth changes the decision 78 % → 30 % of the time,
  not because more is at stake when it does (+0.11 at every stage). The axis that carries it is MATERIAL left, not
  reveal count. DESCRIPTIVE.
- **Where honest search earns its keep: only late and in material-poor positions, and only pooled.** All six seen:
  4-turn leaf over the critic search +0.026 [+0.004, +0.051]; per checkpoint the bounds straddle 0. **No stage's gate is
  open** under the replicated rule (§4). DESCRIPTIVE.
- **How far: roughly an order of magnitude of belief.** Give the true world weight p inside the search's average: the
  gain is concave (half by p ≈ 0.22). The leaf clearly beats the critic search at p\* ≈ 0.05 (late) to 0.20–0.35
  (early); our best honest belief is worth p ≈ 0.01 [0, 0.03]. DESCRIPTIVE.
- **What is missing, by stage:** early, the NEVER-SEEN Pokémon (species and sets; the seen mons' moves explain ~0 of the
  headroom); in the middle, the SEEN mons' unseen moves (share 0.55 at k = 4); items, abilities and spreads are not yet
  measured (the pilot told them true). The network's own belief is barely sharper than Smogon usage.
- **What does not help:** more rollouts as critic-training labels, a per-action head on noisy oracle labels (−0.004), the
  proof-first 2·SE acting rule, trusting the policy prior at Gumbel's default scale, shared-seed dice pairing beyond one
  turn, a privileged critic, telling the frozen network the true moves (+0.001).
- **Open, in order of cost:** the Tier-1 factorial on the fully honest base (registered, no arm read), the replica's
  third-party comparison, the knock-in forks, the maturity battery at plateau, the stage-gated bootstrap's gate (§7).

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
| W9 | **Partial truth in the belief is worth a lot (the dose-response is concave).** Weight p on the true world among K honest ones | PROVISIONAL, DESCRIPTIVE | pooled leaf +0.009 (p = 0) → +0.034 (0.1) → +0.062 (0.35) → +0.069 (0.5) → +0.076 (1.0); half the rise by p ≈ 0.22 [0.17, 0.27]; the true world beats a random honest world at the same weight (+0.012 [+0.009, +0.014] at p = 0.1) | `hidden_info_by_reveal_count_2026-10-12/` README §4 |
| W10 | **An existence proof that determinised search wins here (PRE-ERA, a width meter).** Foul Play (decoupled UCT over a hand-written evaluator, Smogon-determinised sets) vs the 75M policy | MEASURED, PRE-ERA | our win rate **0.388** [0.288, 0.497], n = 80, ~1.4M visits per decision; width and opponent-set knowledge NOT SEPARABLE (width slope +0.027 per ln-visit, CI covers 0; knowledge delta +0.000 [−0.15, +0.15]). Later (PRE-ERA Kakuna, greedy vs Foul Play at 1000 ms): 0.490 [0.422, 0.559], NOT DETECTED | `UNDERSTANDING.md` §3.1; ledger 2026-09-14 L18248, 2026-09-17 L18964, 2026-09-29 L21447 |
| W11 | **A rollout leaf out-ranks the learned critic head (PRE-ERA).** K = 1/2/4/8/16: 0.543 / 0.580 / 0.613 / 0.652 / 0.671 vs 0.574 | MEASURED, PRE-ERA | DETECTED at K ≥ 8 (+0.078 [+0.027, +0.128]); the single-rollout label itself mis-orders the top1\|top2 pair 39.7 % of the time (so any pairwise level is compressed toward 0.5) | ledger 2026-09-19 L19881, L19663 |

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
| N8 | **Telling the NETWORK the true moves** (frozen; and trained) | PROVISIONAL (frozen); MEASURED (X5 oracle arms, below the floor) | injected at 95 %: it flips 13 % of picks and gains **+0.001** [−0.003, +0.005] (R-NET, CRITIC-USES NOT DETECTED, 3 networks); a wrong-moves placebo costs the critic search ≈ 0.8 points. X5 oracle arms: told species +2.44 pp [+0.31, +4.57], full set +2.18 pp [+0.53, +3.84], the set over species −0.26 [−1.92, +1.40], all below the 4.57 pp replicate floor; an untrained network handed a revealed team FELL −8.1 pp | a USABILITY limit, not an information one: the frozen network does not convert the move content. Whether a network can LEARN to is the knock-in forks' question (§7) |
| N9 | **Smogon-usage or network-belief worlds as a sufficient sampler** | PROVISIONAL | R-HONEST NOT DETECTED (+0.011 / +0.007); R-BELIEF (network species belief − Smogon) −0.010 [−0.024, +0.005] / −0.002; the network's MOVE belief closes ~⅕ (end-state) / ~1⁄20 (production) of the oracle gap (R-MOVES +0.017 [+0.001, +0.031] HELPS end-state only, NOT DETECTED on production, builder-sensitive) | the belief is barely sharper than Smogon usage (51 % / 48 % of a world's hidden moves right vs 45 %); a better species hit rate (0.45 vs 0.21) buys NO value (B′ ≈ 0) because value needs the whole world to hang together |
| N10 | **"Focus the search on torn states"** | PROVISIONAL | LEAF-K8 flat across the policy's gap quintiles (q1 +0.078 / +0.090, q5 +0.075 / +0.072); banked: torn 40 % +0.049 vs most decided 20 % +0.082. No cheap policy-confidence feature separates the decisions search flips (PRE-ERA: flip rate 0.694 [0.681, 0.707]) | the stakes are larger where the policy is confident |
| N11 | **Narrowing the state distribution to buy discrimination** (an exploiter's head) | MEASURED, PRE-ERA | sibling ranking 0.575 on its own slice and the generalist's alike | the 0.575 ceiling was four heads deep: coverage, width, loss form and data each varied |
| N12 | **Hand evaluators as the leaf (PRE-ERA)** | MEASURED, PRE-ERA | nine tried; none beat the head; the PBRS potential matches a 75M critic pooled | Foul Play's edge is evaluator, search shape or knowledge: not separated (§7) |

---

## 4. THE FRONTIER MAP (by stage of the game)

**How to read it.** G = oracle IMP − honest IMP: what perfect information would add, measured against the pilot's honest worlds (a LOWER
bound on the fully honest gap O − F, because those worlds told the seen mons' item, ability and spread true). The honest columns are the pilot's honest worlds
(Smogon-prior A′; "vs critic" is honest LEAF-K8 / TRUNC-4-K8 over the honest one-ply critic search, same worlds). Pooled
over both policies, battle-clustered 95 % on G; n 192–351 per row, so read the pattern, not a digit. PROVISIONAL,
DESCRIPTIVE (post hoc, on the original study's rows; the registered arms are `design_hidden_info_binding.md` §12.13).
The three axes are correlated (k–material −0.71, k–turn +0.66), so the rows below are NOT independent evidence.

| stage | n | oracle IMP | honest IMP vs policy | **G (oracle − honest)** | honest vs critic (LEAF / TRUNC-4) | which knowledge is missing | gate |
|---|---|---|---|---|---|---|---|
| **k = 1–2** seen | 325 | +0.081 | −0.004 | **+0.086** [+0.066, +0.104] | −0.008 / +0.001 | the never-seen Pokémon (seen mons' moves share of headroom 0.04 [−0.25, 0.25]) | CLOSED |
| **k = 3** | 192 | +0.065 | −0.016 | **+0.081** [+0.057, +0.106] | −0.021 / −0.008 | the never-seen Pokémon (share 0.09) | CLOSED |
| **k = 4** | 209 | +0.099 | +0.023 | **+0.076** [+0.051, +0.102] | +0.002 / −0.011 | the seen mons' unseen moves (share **0.55** [0.27, 0.81]) AND the never-seen | CLOSED |
| **k = 5** | 228 | +0.068 | +0.012 | **+0.056** [+0.036, +0.078] | +0.003 / +0.009 | mixed (share 0.30 [−0.15, 0.61]) | CLOSED |
| **k = 6** | 232 | +0.066 | +0.031 | **+0.035** [+0.017, +0.055] | +0.016 / **+0.026** | nothing unseen: the whole headroom (+0.021) is the seen mons' moves (O − CT′ = 0.000 on all 232 states, an identity check) | WEAK (pooled only) |
| material **≥ 12** alive | 231 | +0.078 | −0.009 | **+0.087** [+0.066, +0.108] | −0.013 / +0.002 | the never-seen (share 0.00) | CLOSED |
| material **10–11** | 351 | +0.086 | +0.001 | **+0.086** [+0.066, +0.105] | −0.022 / −0.007 | never-seen first (share 0.24 [0.01, 0.44]) | CLOSED |
| material **8–9** | 260 | +0.091 | +0.018 | **+0.073** [+0.052, +0.096] | +0.023 / +0.017 | seen mons' moves (share 0.43 [0.14, 0.64]) | CLOSED (points above 0; per-checkpoint bounds in `tables.md`) |
| material **≤ 7** | 344 | +0.054 | +0.022 | **+0.031** [+0.018, +0.045] | +0.009 / +0.007 | little left | CLOSED (beats the policy, not the critic search) |
| turn **≤ 7** | 299 | +0.083 | −0.008 | **+0.091** [+0.071, +0.111] | −0.016 / −0.001 | the never-seen (share 0.01) | CLOSED |
| turn **8–14** | 328 | +0.080 | +0.014 | **+0.066** [+0.048, +0.085] | +0.010 / +0.008 | mixed (0.26 [−0.01, 0.49]) | CLOSED |
| turn **15–21** | 272 | +0.086 | +0.012 | **+0.074** [+0.053, +0.097] | −0.005 / +0.007 | seen mons' moves (share 0.54 [0.25, 0.76]) | CLOSED |
| turn **≥ 22** | 287 | +0.055 | +0.016 | **+0.039** [+0.022, +0.056] | +0.005 / +0.001 | little left | CLOSED |

**What the table says.**
1. **Frequency, not stakes.** G = c × s exactly, c = how often the oracle's pick differs from the honest pick, s = the
   true-value gain when it does. s is flat (0.104–0.115; early − late +0.001 [−0.037, +0.038]); c falls 0.78 (k ≤ 2) → 0.30
   (k = 6) (early − late +0.352 [+0.291, +0.413]).
2. **Material is the axis.** Regressed on k, turn and material together, only material keeps its slope (k −0.002
   [−0.017, +0.012]; material **+0.022** [+0.010, +0.035] per SD, pooled). At equal material (≥ 8 alive) k ≤ 3 and k ≥ 5
   have the same gap (+0.013 [−0.019, +0.047]). By alive count: +0.073 / +0.086 / +0.087 (8–9 / 10–11 / 12) vs +0.031 (≤ 7).
3. **The oracle's edge is flat (+0.065…+0.099) at every stage; the stage structure lives on the HONEST side.**
4. **The frontier today** is the late, material-poor corner, which is also where there is least to gain (G +0.035 at k = 6,
   +0.012 [−0.002, +0.030] on production alone: nothing left to find). The cheapest labels sit in the stage with the
   least headroom (`design_value_discrimination.md` §13.10).
5. **Gate status.** The registered gate (R-STAGE-SEARCH, `design_hidden_info_binding.md` §12.13.4: honest TRUNC-4-K8 beats
   BOTH the policy and the critic search, lower95 > +0.0005, on EACH checkpoint separately and pooled at 99.6 %,
   n ≥ 150 per checkpoint) has NOT been run. Read descriptively, **no bucket is open per checkpoint**; k = 6 would pass
   pooled only (GATE-WEAK), and the one-ply critic search alone beats the policy on end-state in some late buckets
   (k = 5 +0.023 [+0.001, +0.047]; turns 15–21 +0.035; ≤ 7 alive +0.028) but in none on BOTH checkpoints.

---

## 5. HOW FAR WE ARE

### 5.1 The belief-quality dose-response

Search averages K worlds; the true world gets weight p, the honest worlds share 1 − p. One set of banked per-world
outcomes gives the whole curve by re-weighting (no playout was run). **p is a weight on the exact true world, NOT a
probability the network holds** (no real belief puts mass on a whole team); "today's p" is the p-EQUIVALENT in value
units: the p at which the curve equals a world kind's own measured IMP.

| | p = 0 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| leaf K8 | +0.009 | +0.034 | +0.040 | +0.062 | +0.069 | +0.078 | +0.076 |
| 4-turn leaf | +0.014 | +0.021 | +0.033 | +0.040 | +0.044 | +0.043 | +0.044 |
| one-ply critic search | +0.010 | +0.010 | +0.010 | +0.011 | +0.014 | +0.018 | +0.019 |
| leaf, literal K = 5 (one playout per world) | +0.007 | +0.018 | +0.025 | +0.032 | +0.032 | +0.032 | +0.028 |
| same with a WRONG (random honest) world in the true one's place | +0.007 | +0.006 | +0.008 | +0.006 | +0.006 | +0.005 | +0.003 |

- **The shape:** concave. The first 10 % of truth buys 37 % of the oracle's rise, 35 % buys 79 %, 50 % buys 90 %. The
  critic cannot use the extra information (+0.010 → +0.019); a rollout can. A confidently wrong world helps at no weight.
- **Where we sit (p-equivalent, pooled):** the network's own belief worlds (C′) **0.010** [0.000, 0.029]; its species
  belief (B′) 0.000; even a world that knows every seen mon's TRUE set (CT′) only **0.045** [0.027, 0.19].
- **The break-even p\*** (smallest p after which the leaf clearly beats the critic search at the same p; 95 % lower
  bound): pooled 0.05; by k: 1–2 **0.20**, 3 **0.35**, 4 0.05, 5 0.25, 6 0.05; by turn: ≤ 7 0.20, 8–14 0.05, 15–21 0.05,
  ≥ 22 0.25; by alive: ≤ 7 0.05, 8–9 0.05, 10–11 0.20, ≥ 12 0.35. Half the rise (p-half): 0.23–0.31 early and
  material-rich, 0.16–0.25 in the middle, 0.04 at k = 6.
- **So:** early and material-rich stages need ≈ **0.20–0.35** of truth-weight, **20–35× today's ≈ 0.01**; late stages need
  ≈ 0.05, a few times today's. Intervals are wide (n ≈ 200–330 per stage): read the pattern.
- **Units caveat (§4.3 there).** In hit-rate units the picture differs and the two bridges disagree: species hit rate
  0.45 vs Smogon 0.21 ⇒ p_eff ≈ 0.30; a seen mon's move SET recall 0.51 / 0.48 vs 0.45 ⇒ 0.11 / 0.065; one move slot
  0.187 / 0.172 vs 0.166 / 0.160 ⇒ 0.026 / 0.015. The better species hit buys no value, so hit rates overstate partial
  correctness; the registered rules use value units.
- **Not yet measured:** TODAY on the FULLY honest base (the FN arm is a build item); every number above is on A′ (the
  pilot's builder). Stage-by-stage p\* are post hoc hints with no multiplicity adjustment.

### 5.2 The value of information by CLASS

| class of hidden fact | what we know | value | tag |
|---|---|---|---|
| **never-seen Pokémon** (identity and sets) | oracle − CT′ = the never-seen mons plus what the pilot kept true: +0.039 (end-state) / +0.053 (production); at k ≤ 3 the whole headroom (+0.08) with the seen mons' moves explaining +0.003–0.007; the model-input axis cannot test it (the network is told seen mons' moves only) | the largest EARLY, and the one no per-fact hit rate we have measured reaches | PROVISIONAL, DESCRIPTIVE |
| **seen mons' unseen moves** | CT′ − C′ = +0.024 [+0.011, +0.039] (end-state, HELPS) / +0.010 [−0.003, +0.023] (production, not clear); share of headroom 0.55 [0.27, 0.81] at k = 4, 0.54 [0.25, 0.76] at turns 15–21; the whole headroom at k = 6 (+0.021). Post hoc: with all six seen, the gap vanishes once ≥ 10 of the seen mons' 24 moves are revealed | the middle and late stages; the network's move belief reaches p ≈ 0.01 only | PROVISIONAL |
| **items, abilities, spreads, exact HP** | NOT MEASURED: the pilot told a seen mon's item, ability and spread true. Tier 2 of the study measures F+I/A/S and O−I/A/S | unknown; bounded above by the pilot-vs-honest gap in §5.3 | UNVERIFIED |
| **hidden counters, speed ties** | residual leaks of the honest builder (§6.1) | unknown | UNVERIFIED |

The apportionment of the gap among classes is the study's Tier 1 (`design_hidden_info_binding.md` §6, §12.3); the author's
pre-factorial confidence that the seen mons' MOVES are THE binding constraint was ≈ 15 %, that hidden TEAM information binds
search ≈ 85 % (`design_hidden_info_binding.md` §16.6). Nothing from the study may steer a multi-week programme until the
replica agrees, the red team has attacked it and the plateau replication confirms it (`design_hidden_info_binding.md` §10.5).

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

**Consequences.** (a) Every "honest" IMP in §2–§4 (A′, B′, C′, CT′) is an UPPER bound on a fully honest search (F): the
honest gap G in §4 is the pilot's M + U gap, a LOWER bound on the full value of information O − F. (b) The conclusions that
search does NOT beat the critic are therefore conservative; the late-game edge (W6) is the claim most at risk of shrinking.
(c) The fully honest numbers do not exist yet: they are Tier 1's F arm. The pilot's archived worlds and rows are
unaffected (they were built by the pinned twin `determinize.py`).

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
  facts, `playout_open`); its third-party comparison is OPEN (§7).

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
Rust side, not built); the training-time `--belief-oracle` of the study's §8.1 is NOT built.

---

## 7. OPEN QUESTIONS, and the experiment that settles each

| # | question | the experiment | cost / status |
|---|---|---|---|
| Q1 | Does a FULLY honest search beat the critic anywhere, and which class of fact carries the gap? | **Tier 1 of the hidden-info study** (`design_hidden_info_binding.md` §6, §12.13): arms O, F, F+M, O−M, F+U, O−U, FN on the k-stratified Tier-1S sample (600 registered + 420 supplement states per checkpoint); rules R-STAGE-VOI / -AXIS / -MECH / -CLASS / -SEARCH / R-DOSE; Tier 2 adds items, abilities, spreads | ≈ 58–65 CPU-h for the 9 arms × 2 checkpoints (re-priced from §15's 47), CPU only. REGISTERED (Amendment 2), no arm read at the writing of this ledger. FN (the network's belief on the F builder) is a build item (~0.5 agent-day, UNVERIFIED) |
| Q2 | Is the pilot's oracle − honest gap an artifact of our instrument? | **The blind replica's third-party comparison** (R-REPLICATE, then the red team, `design_hidden_info_binding.md` §13) | PENDING. Not read here. Its three relayed findings are answered (§6.1) |
| Q3 | Can a network LEARN to use told information, and what is each class worth in strength? | **The knock-in forks**: `--oracle-reveal ki_moves` (etc.) with `--diagnostic-fork-consent`, 5 arms × 3 seeds × +3M from the adopted end state, a no-change control (the v8 guard) and a wrong-moves placebo; `main.dose` for the dose match | ≈ 14 GPU-h; QUEUED behind the bf16 A/B. The study's §8.3 argv is NOT launchable as written (`--arch endstate` is refused on a fork; `--belief-oracle` not built) |
| Q4 | What do maturity and plateau change: does G shrink, does today's p rise toward p\*, does the frontier move earlier? | **The maturity battery** on the X26 deep run (`design_hidden_info_binding.md` §12.13.7): O, F, FN, F+M, F+U re-run at ≥ 25M then every 25M and at the plateau checkpoint, ONE frozen 800-state bank, rules R-MATURITY (GAP-SHRINKS / BELIEF-RISES / FRONTIER-MOVES-EARLIER / -STATIC / -RECEDES) | ≈ 150 CPU-h for six checkpoints (FULL ≈ 47 at the first and plateau anchors, LEAN ≈ 14 between), no GPU. Every claim in this ledger is PROVISIONAL until it runs |
| Q5 | Does the honest ONE-PLY critic search pay in GAMES? | **S0b** (`design_value_discrimination.md` §13.4): the end-state checkpoint with an 8-world one-ply critic search vs the same checkpoint raw, mirrored head-to-head; pass = lower bound > the +4.57 pp replicate floor | ≈ 1 CPU-day. NOT run as far as the ledger records; OWNER approval requested (§13.0 ask 2) |
| Q6 | Is the stage-gated bootstrap viable: does honest search work anywhere well enough to label? | **R-STAGE-SEARCH** (Tier 1S, 13 buckets, per checkpoint and pooled at 99.6 %), then the S1 SHADOW (logs only, per-stage `search/imp`), then the pre-registered A/B (`design_value_discrimination.md` §13.4, §13.7, §13.10) | A DESIGN PROPOSAL for the owner's review, nothing built. Gate closed today (§4). The in-training deferral (2026-09-30) is NOT lifted by this doc |
| Q7 | The oracle-critic roles: may an oracle-informed critic or oracle labels help? | (a) the offline omniscient twin X11 as a measuring instrument (V − E_belief[V\*]; where is error reducible, where is information worth gathering), never a target; (b) oracle playouts as value labels for a TRAINED body or a much larger label budget (`design_value_discrimination.md` §13.8, GPU; needs the owner's ruling on in-training search). The 09-10 `truevalue` arm (N7) argues against training the public V on true values | not run; (b) is the only route that reaches the hidden-information-limited label |
| Q8 | Is the move-belief gap closable at all? | the dose-response's TODAY on the F base (Q1's FN arm) against p\*; a SHARPER belief than the network has (the CT′ ceiling beats the critic on end-state); the model-input axis already says the network does not convert told moves (N8) | rides on Q1 and Q3 |
| Q9 | Is the evaluator or the information the limit (Foul Play's hand evaluator vs our critic)? | **R-EVAL**: the hand evaluator (BUILT, `src/main/hidden_info_study/hand_eval.py`) as a second leaf in the factorial | its playout arm runs with Tier 1 |
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
