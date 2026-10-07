# X5: discrete fixed-mass belief tokens + OTHER, design note and A/B pre-research

**Status: REVISED after independent review (2026-10-03); M2 / M3 decided by the owner (§9, Decision record). Build
units U1 (the dex-row table), U2 (the T0 hypothesis builder, the `--belief-tokens` flag), U3 (parts 1–3: the class-E
half, the opponent active's move axis, the op's opponent-MON axis and OTHER's physics — §8.3, U3 part-3 hand-off),
U4 (the flat opponent pointer, OTHER labels, the re-expressed cells, the B re-base — §3.7 "As built (U4)", the U4
hand-off) and U7 (the readers: `main.belief_roles`, §4.3 / §7.4 "As built (U7)", the U7 hand-off) are BUILT; the rest
is not.** The first version (`00cdf0c2`, decisions `81578969`) was reviewed SOUND WITH FIXES with ten must-fix items
(M1–M10). This revision resolves each one: M1 and M4–M10 by orchestrator decision or measurement, M2 and M3 as two real
choices for the owner. Each item names where it landed (§8.4).

This note is the build spec that EXPERIMENT_BACKLOG row X5 asks for. It refines [`design_q_head.md`](design_q_head.md)
§1 (the tokens) and §3 (the flat opponent pointer). Where it departs from that spec, §3.9 says so and gives the reason;
that spec carries a pointer here. It records the KL early stop's analysis (§5, NOT built: owner, OFF), plans the K9
golden re-bake (§6) and registers the A/B (§7).

**Standard (owner, 2026-10-03).** "I don't want slop entering our core."
- Every design choice cites its literature, or says plainly that it is novel and names the risk.
- No homebrew where a grounded method exists.
- Every check is deterministic (standing rule 8).
- Open gaps are listed (§8.2), not hand-waved.

**Evidence tags.**
- MEASURED: banked, with a pointer.
- ESTIMATED: arithmetic on measured figures, with the assumption stated.
- UNVERIFIED: not checked.

Ledger lines are `L…` keys from [`../research_state/ledger_index.md`](../research_state/ledger_index.md). The revision's
own measurements are banked in [`../research_state/measurements/x5_revision_2026-10-03/`](../research_state/measurements/x5_revision_2026-10-03/)
(scripts + outputs, seeded, reproduced exactly).

---

## 0. Summary

1. **What is wrong today (MEASURED, §1.2).**
   - An unrevealed opponent mon is a learned per-position CONSTANT token. No species information reaches it.
   - Its move, item, HP-type and spread posteriors are therefore state-independent on hidden slots, apart from the
     parameter-free E10 move mixture.
   - The damage op prices a hidden DEFENDER on AVERAGED stats, E[def], E[spd], E[maxhp], which is a Jensen gap. It does
     not price a hidden ATTACKER at all.
   - The belief readout memorises the pool: ECE 0.26 on-pool against the Smogon prior's 0.017.
   - The opponent-intent head has no label on 38–42 % of decisions. That figure is an upper bound on the belief-miss
     share.
2. **X5 replaces the constant tokens with concrete hypotheses.**
   - Each unrevealed seat holds a real species hypothesis (top by presence), with presence π.
   - π is the **logistic fixed-size marginal** π_s = σ(a_s + τ), with a_s = log P_T0(s | revealed) + δ_θ(s) and τ set
     by a deterministic bisection so that Σπ = k = 6 − r (§3.2). This is the entropy-regularised projection onto the
     k-subset polytope (Amos, Koltun & Kolter 2019, LML) and the I-projection of independent Bernoulli presences onto
     "expected count = k" (Csiszár 1975). π never reaches 0 or 1, so its BCE is finite with a live gradient
     (review M1: the earlier capped πps gave π = 1 and an infinite BCE on 0.23 % of cold-start decisions, MEASURED).
   - ONE new token, OTHER_species, holds the leftover mass. The opponent active's moves follow the same scheme, with
     mass 4 and OTHER_move re-using the active's existing E5 tail seat.
   - π is DETACHED wherever it weights the policy or critic (§3.2, M10): δ_θ learns only from the presence BCE.
3. **Presence enters every EXPECTATION-type reduction over opponent tokens as a log-weight bias** (ToMe's
   "proportional attention", Bolya et al. 2023). Two deterministic invariances pin those reductions: zero presence
   equals masking (I1) and splitting a token into two halves equals the token (I2). **The 14 MAX- or TOP-K-type
   reductions cannot satisfy both** (review M2; all 41 opponent-axis reductions are enumerated in §3.5). Their
   semantics is an owner choice (§9 M2); the recommendation keeps today's presence-scaled max, which adds no second
   lever.
4. **Hypotheses get real physics; OTHER's physics is an owner choice (§9 M3).**
   - Each hypothesis seat's dex row is produced by THE observation encoder, as a committed artifact behind a parity
     gate and a real-state cross-check (§3.4).
   - The op prices hypotheses as concrete defenders AND attackers.
   - **OTHER carries 57–74 % of the unseen mass at a 1:1 hypothesis budget** (r = 1–5), under the Smogon prior and
     under every pool-memorising proxy we could build; top-k recall of the true hidden mons is 0.34–0.52 (MEASURED,
     §3.3). With no physics for OTHER, X5 would price NONE of that mass where today's blob prices all of it: a
     regression on the physics channel. The recommendation (§9 M3, option c) gives OTHER today's averaged construction
     restricted to the tail, so X5 becomes a strict refinement of the blob.
5. **The opponent pointer becomes one flat list.**
   - Candidates: the move seats, OTHER_move, each revealed bench mon, each hypothesis, and OTHER_species.
   - A belief miss becomes an OTHER label instead of a masked row.
   - β and the content-addressed target are retired; the ride-along B head is re-based onto the flat list.
6. **Cost (§3.6).**
   - Sequence 61 → 62 tokens at the recommended budget.
   - About +1.2 % forward matmul FLOPs and parameters roughly neutral (ESTIMATED).
   - The op's per-hypothesis attacker physics is NOT costed; build unit U8 measures it against a pre-registered budget.
   - No Rust runtime change.
7. **The KL early stop is OFF** (owner, 2026-10-03): not in the A/B, not in X26, not built in the X5 units. §5 keeps
   the fire-rate analysis as input to X28, the epoch controller that is now the named lever after X26.
8. **Run variance dominates the A/B (§7.1).** The replicate floor is about 98 % training-run variance. σ_run on the
   untaught meter is 3.43 pp from the seed pair, but the same-seed cross-pin pair A − A2 reads −11.35 pp (σ ≈ 8.0 from
   that pair alone; pooled over the three seed/pin pairs 5.2; SD of the four E10 arms 4.8). Every sizing table carries
   σ ∈ {2.5, 3.43, 4.8} rows (review M4).
9. **The A/B, registered (§7.4; orchestrator decisions, review M5–M8).**
   - **Primary meter: the mirrored HEAD-TO-HEAD cross** between every X5 seed's final snapshot and every blob seed's.
     Pre-registered now; P0 is a planning input only. The untaught meter is a reported secondary.
   - **Margin δ = 3.5 pp** (≈ 24 Elo), one-sided α = 0.05.
   - **Group-sequential with estimated variance:** looks at 3, 5 and 8 seeds per arm (8 ≈ 41 GPU-h, the owner's
     ceiling); O'Brien–Fleming-shaped boundaries on the t scale, t ≥ 5.761 / 2.683 / 1.874 on 4 / 8 / 14 df; a
     non-binding futility stop at looks 1–2 when the estimate is at or beyond −δ.
   - **Simulated (MEASURED by simulation, §7.4):** type-I ≤ 0.047 at every σ; **power at Δ = 0 is 0.81 / 0.57 / 0.36
     at σ = 2.5 / 3.43 / 4.8**, and 0.33 at the pooled 5.2. **At σ = 3.43 the design is UNDERPOWERED (0.57 < 0.6) even
     at the 41 GPU-h ceiling.** δ = 4.5 pp (≈ 31 Elo) gives 0.94 / 0.76 / 0.51 at the same cost. Margin against budget
     is the owner's call (§9).
   - Expected GPU-h ≈ 34 if X5 is truly equal, ≈ 26 if it is truly δ worse.
   - The outside panel is REPORTED, never gated, with a pre-declared harm flag (§7.4). The purpose-metric adoption gate
     infers across seeds (per-seed statistic, t over seeds).
10. **Build (§8.3): about 11–13 agent-days, every unit Opus-tier**, after the deletion pass and before X26. U5 (the KL
    stop) is dropped. §9 holds the owner's two brainstorm choices (M2, M3) and the margin-against-budget trade.

---

## 1. Problem statement

### 1.1 What X5 is

Backlog row X5, option 2: **discrete per-mon hypothesis tokens with FIXED total mass, plus an OTHER token**.
- **Their team:** revealed mons at weight 1, species hypotheses for the unseen remainder, and OTHER_species. The group
  mass is 6.
- **Their active's moves:** revealed moves pinned at 1, move hypotheses, and OTHER_move. The group mass is 4.
- Hypotheses carry presence probabilities, not slot assignments. OTHER's mass is the calibrated "how blind are we"
  quantity, so it is never renormalised away.

It adds no ply, so it stays in scope under the owner's one-ply rule. It is the North Star 1 retrain boundary: the X26
baseline continues from whichever arm this A/B selects.

### 1.2 The blob as built (`900de5e2`, verified in code)

| piece | what it is today | pointer |
|---|---|---|
| **T0SpeciesPrior** | Parameter-free Smogon naive Bayes, `log P(s) + Σ_r log-lift(s, r)` over the chaos `Teammates`. Species Clause is a hard override at a FINITE logit (`SPECIES_CLAUSE_LOGIT` = log 1e-6). It produces ONE team-level categorical `[B, S]`: "the species in a hidden slot". It carries no count k and no without-replacement step. The marginal has a floor at 1e-4 and the lift is clipped to ±4 nats. | `t0_species.py:42-118`, `belief_tables.py:498-540` |
| **BeliefSlots** | Each unrevealed opponent seat gets one of 6 learned per-position CONSTANT tokens, `unknown_slot_emb [6,128]`. No species information goes into the token. | `belief_heads.py:24-61` |
| **MoveBelief (hidden slot)** | E10 mixture `logit(P_T0 @ P(m\|s))` plus `move_head(token)`. The token is constant, so the learned part is a state-independent per-slot constant. Outputs are independent sigmoid inclusion probabilities over the 400-move vocabulary (learnset-gated prior, revealed moves pinned at logit 10). | `belief_heads.py:265-420` |
| **Item / HP-type / spread (hidden slot)** | Read the constant token plus the species-0 prior row. They are state-independent, and they are never supervised there because labels exist only for revealed slots. | `per_slot.rs:340-348`, `spread.rs:185-199` |
| **E4 threat seats** | Top-6 of the opponent ACTIVE's candidate moves by sigmoid weight. The index is detached; `w` is differentiable. The top-6 is computed TWICE (`damage_op_blocks.py:875` for the op, `:961` for the seats) with `torch.topk`'s unspecified tie order. | `pointer_head.py:114-138`, `damage_op_blocks.py:936-962` |
| **E5 tail seats** | One per opponent mon, `[p_tail, worst_phys, worst_spec, revealed]`: `p_tail = min(1, Σ tail w)`, tail membership by `w ≥` the 6th value (a different tie rule from E4's), `worst_* = max over the tail of w·BP/150·acc`. No edge bias, no presence term. | `pointer_head.py:139-160` |
| **Damage op, hidden DEFENDER** | Priced on an AVERAGED defender: `E[def]`, `E[spd]`, `E[maxhp]` = `P_T0 @ tables`, applied before the nonlinear damage formula. Only the type multiplier is averaged per species. P(KO) is zeroed and full HP assumed. | `damage_op_blocks.py:351-422` |
| **Damage op, hidden ATTACKER** | Gated OFF (`att_gate` = revealed only), and a hidden slot's encoded HP is exactly 0, so an undeclared `hp > 0` gate also drops it from `p_pur_vs_us`, the v / t / g edges, the speed gate and s1. | `damage_op_pairwise.py:283-287, 974-980` |
| **BeliefHead** | A training-only aux on post-transformer opponent tokens: species = T0 prior + zero-init delta, per-slot CE after an exact min-cost slot matching. It is also β's content-addressed target. | `belief_heads.py:64-199`, `training/belief_bank.py:312` |
| **HiddenOppBeliefPool** | k = 6 DETR-style queries over the 12 team tokens, 768-d. It feeds the policy projection, and also the critic through `value_entity_pool_full`. | `pools.py:118-169`, `extractor_forward.py:768-769` |
| **α / β intent** | α: 6 E4 seats + SWITCH; a move outside the seats is `INTENT_IGNORE`. β: pointer over the 6 team tokens; a hidden target is resolved through BeliefHead's argmax if it clears 0.05, else masked. | `opp_intent.py:57, 97, 129, 269` |

The sequence the TeamTransformer actually runs under production is **S = 61 tokens**: 13 base + 16 entity seats + 32
event-window seats, measured by a forward hook. It is not the 29 or 36 the docs state (§8.2, F-X5-1).

### 1.3 Evidence that this is the problem

- **The belief memorises the pool and is overconfident even on it.** MEASURED; L21301,
  `measurements/belief_calibration_2026-09-24/`; `ai_v13_22_popr1_loop` @103.2M, 57,238 decisions.
  - Species NLL, head vs Smogon prior: pool 2.55 vs 2.88, ladder 4.49 vs 3.21, procedural 5.90 vs 3.86.
  - ECE: head 0.260 / 0.395 / 0.471 against the prior's 0.017 / 0.032 / 0.111.
  - Confidently wrong (top-1 > 0.8 on a species the opponent does not have): 0.095 on-pool. The prior's rate is 0.
  - On-pool, the head's edge over the prior is gone by k ≥ 4 reveals.

  The owner's rule is that memorising the pool is a SUCCESS milestone. The miscalibration, though, is on-pool too.
- **The hidden-slot move posterior was a state-independent constant** (maximum deviation 0.0 over 57k decisions,
  L21301). E10 (`b0a28b5b`) fixed its PRIOR half without learned parameters. Its learned delta still sits on the
  constant token (§1.2).
- **The beliefs matter to play.** MEASURED; L21381, `measurements/belief_winrate_ab_2026-09-24/`. Zeroing the learned
  belief deltas costs −5.2 pp [−7.4, −3.0] against pool teams and −0.4 [−2.5, +1.7] against ladder teams. The
  difference-in-differences is +4.8 [+1.9, +7.8].
- **The intent head misses.** MEASURED by the evidence read on 2026-10-03 from TensorBoard; not banked.
  - `opp_intent/alpha_mask_rate` is 0.420 at 7M and 0.389 at 75M (`ai_v14_01_base`), and 0.378 at 91.1M
    (`ai_v14_07_g0p_k2`).
  - That rate is `1 − n_supervised/B`. It folds in non-choices (drags, forced replacements), so it is an UPPER BOUND on
    belief misses. No banked metric isolates the miss share.
  - Old architecture (L00284): the K = 6 seats held the clicked move 89.3 % of the time, and 46.4 % of switches
    brought in an UNSEEN mon. The current α has no candidate for such a switch; only β's content address covers it.
- **The averaged-blob physics harm is UNMEASURED.** No banked read isolates the Jensen gap or the attacker gate (F-X5-2).
  The design principle is owner-stated (`design_q_head.md` §0). The closest evidence cuts the other way:
  - The privileged `truevalue` critic, which saw the true opponent team, bought "NO resolution and NO opponent
    conditioning" (L17103).
  - So X5 is not expected to move V much. Its case is intent, calibration and the downstream Q/search pointer.

  The A/B is therefore a NON-INFERIORITY test on strength with superiority reads on the purpose metrics (§7), not a
  superiority bet on strength.

---

## 2. Literature review, applied to our setting

All entries were checked by the literature reads of 2026-10-03, by abstract, PDF or source code. The revision's read
corrected five attributions (Hu 2021, Patterson 2024, DPFRL, Gupta & Ramdas 2022, Agarwal 2021) and added the
presence construction's and the group-sequential design's sources. A row marked PARTIAL had one claim not confirmed
from the primary source; "via a secondary source" says which.

### 2.1 Set and slot representations

| work | what it gives | for X5 |
|---|---|---|
| Zaheer et al. 2017, *Deep Sets* (NeurIPS; arXiv:1703.06114) | Permutation-invariant f = ρ(Σφ(x)) | **Adopt for the learned delta's input:** a sum-pool over the REVEALED opponent set (§3.2). It is also the formal description of today's averaged defender (a weighted sum-pool before a nonlinearity), which X5 replaces for the hypotheses. |
| Lee et al. 2019, *Set Transformer* (ICML) — PARTIAL | Attention over sets; PMA pooling by learned seeds | No new block is needed: hypotheses join the existing self-attention. PMA is the description of our CLS pools. |
| Locatello et al. 2020, *Slot Attention* (NeurIPS) | Slots DISCOVER objects through competitive attention | **Reject.** Our entities are given and enumerable (species); there is nothing to bind. |
| **Bolya et al. 2023, *Token Merging* (ICLR; arXiv:2210.09461), §3 Eq. 1** | "Proportional attention": `softmax(QKᵀ/√d + log s)`, where s is each token's size | **Adopt as THE citation for the log-presence bias.** The identity `softmax(l_i + log w_i) = w_i e^{l_i} / Σ_j w_j e^{l_j}` is elementary; ToMe is its published use for "one token = w copies". Its ablation (Tab. 1f) found it necessary for supervised models, not for MAE pre-training. |
| Ying et al. 2021 (Graphormer, NeurIPS); Press et al. 2022 (ALiBi, ICLR) | Additive attention-logit biases, learned (Graphormer) and fixed (ALiBi), train well | Precedent that a FIXED additive bias is benign. Our edge families are already this mechanism. |

### 2.2 Particle and mixture beliefs in POMDP RL

| work | for X5 |
|---|---|
| Ma et al. 2020, *DPFRL* (ICLR; arXiv:2002.09884), §3.4 | **The closest precedent:** a weighted particle belief summarised for the policy as the weighted MEAN particle plus m moment-generating-function features Σ_i w_i exp(v_jᵀh_i) at learned points v_j. X5 feeds the same kind of object, a weighted discrete hypothesis set, through attention with a log-weight bias, which is also a w-weighted sum of exponentials of a learned score. Cite as related. |
| Ma et al. 2020, *PF-RNN* (AAAI); Karkus et al. 2018, *PF-Net* (CoRL) | Supports "a weighted set of discrete hypotheses beats one vector". We need no learned transition or resampling: our hypothesis space is enumerable and its prior is Smogon. |
| Igl et al. 2018, *DVRL* (ICML) | **Reject the machinery** (an SMC ELBO trains the belief). Our prior is known and our labels are exact (the true team), so plain supervision is the grounded choice. |
| Silver & Veness 2010, *POMCP* (NIPS); Hájek 1964; Chen, Dempster & Liu 1994 (Biometrika 81(3):457–469) | For the LATER search phase: conditional-Poisson (rejective) sampling, the maximum-entropy fixed-size design, turns X5's π into joint team worlds. Hájek (1964, Thm 5.2, via a secondary source) gives inclusion probabilities equal to the working probabilities up to a relative O(1/d) error, d = Σπ(1 − π); Chen, Dempster & Liu give the iterative correction to hit the targets exactly. Not part of X5. |

### 2.3 Entity RL and pointer heads

- Vinyals et al. 2015, *Pointer Networks*.
- AlphaStar (Vinyals et al. 2019, Nature): entity transformer plus pointer selection. The entity details were confirmed
  from secondary sources only.
- OpenAI Five (Berner et al. 2019, arXiv:1912.06680): target chosen by dot-product attention over unit embeddings.

All three ground the **flat opponent pointer over a variable candidate list**. Masked candidates are unrepresentable, as
in ours. **Adopt.** OpenAI Five's cross-replica max-pool is not adopted, because hypotheses are alternatives, not
replicas.

### 2.4 Beliefs in imperfect-information games, and Pokémon

| work | what it does with hidden info | for X5 |
|---|---|---|
| **Moravčík et al. 2017, *DeepStack*** (Science; arXiv:1701.01724) | The value network INPUT is both players' RANGES: probability vectors over private hands, bucketed. | **The strongest precedent** for an explicit belief vector as network input. Top-k + OTHER is a truncated range with a remainder bucket, the same move as DeepStack's bucketing. |
| Brown et al. 2020, *ReBeL* (NeurIPS) | Public belief states condition the value. | Same lesson; points toward search later. |
| Hu, Lerer, Brown & Foerster 2021, *Learned Belief Search* (arXiv:2106.09086) | In Hanabi, a supervised, autoregressive learned belief (an LSTM encoder and a decoder over cards, trained by maximum likelihood to predict the player's own hidden hand from the action-observation history) replaces exact beliefs, recovering 55 % of exact search's gain at 35.8× less compute (91 % at 4.6×). | **Adopt the training principle:** X5's delta is a supervised belief head on the true team (§3.2). |
| He et al. 2016, *DRON* (ICML) | An auxiliary opponent model inside the RL network. | Where α belongs: kept as an aux pointer under the existing gradient mode. |
| Perolat et al. 2022, *DeepNash* (Science) | Model-free equilibrium play with NO explicit belief. | The counter-position. Not adopted: our owner direction is discrete-first, and Q and search need an explicit opponent candidate list. |
| Wang 2024, MIT MEng thesis (`../references/wang2024_pokemon_rl.pdf`) | Random battles: MCTS samples ONE filling per trajectory from Showdown's random-team generator; the network sees "unknown" flags. | The generator prior does not exist in OU. Confirms the representation problem is open. |
| **Grigsby et al. 2025, *Metamon*** (arXiv:2504.04395; RLC 2025 per its README) | The observation holds ONLY the opponent's active mon. The team is inferred implicitly from memory over the trajectory. Replays are reconstructed with a human-team model. | **The main opposing design** (implicit belief). We keep explicit beliefs because Q and search need concrete opponent columns (`design_q_head.md` §3–§6). Metamon's team model is a precedent for a teammate-conditioned prior. |
| **Foul Play** (pmariglia; code at `6c467c0`, read) | Samples N determinized worlds at weight 1/N. Unrevealed mons are added ONE AT A TIME by mean teammate co-occurrence over the revealed mons (top 50). Smogon MOVE usage is treated as a per-mon INCLUSION rate (`1 − (1 − r)^{1/n}`). | Its teammate-conditioned prior is our T0. Its sampling does not respect the fixed set size; the author calls it "very non-scientific" for gen-3 OU. X5's fixed-mass π is the principled version. Move usage being an inclusion rate grounds §3.2's move side. |
| Karten et al. 2025, *PokéChamp* (ICML) | Point estimates of unknown stats from usage; an LLM predicts opponent actions. | Point estimates, not a distribution. The LLM's action predictor parallels α. |
| Yu 2026, *PokaiTrainer* (arXiv:2608.29197) — PARTIAL | CFR over public belief states in VGC. | Its belief representation was not confirmed from the abstract. Read in full before X4's return (§8.2). |

### 2.5 Fixed-size presence, open sets, calibration

| work | for X5 |
|---|---|
| **Amos, Koltun & Kolter 2019, *The Limited Multi-Label Projection Layer*** (arXiv:1906.08707; preprint) | **Adopt (§3.2).** LML projects scores x onto the k-subset polytope with a binary-entropy regulariser, `argmin_{0<y<1} −xᵀy − H_b(y) s.t. 1ᵀy = k` (Eq. 1). The solution is **y = σ(x + ν\*)** (Eq. 3), ν\* the root of Σσ(x + ν) − k (Eq. 4), with an implicit-differentiation backward (§3.2). Their solver is a parallel bracketing generalisation of bisection (d = 100 points per iteration on GPU, initial bracket [−x_(k) − 7, −x_(k+1) + 7]); ours is a plain fixed-step bisection on a provable bracket (§3.2). |
| Csiszár 1975, *I-divergence geometry …* (Ann. Probab. 3(1):146–158) — PARTIAL (abstract) | The second reading of the same construction: −xᵀy − H_b(y) equals Σ_i KL(Bern(y_i) ‖ Bern(σ(x_i))) up to a constant, so π = σ(logit p + τ) is the I-projection of independent Bernoulli(p_i) presences onto "expected count = k", an exponential tilt. The theorem number is not checked; cite the paper, not a section. |
| Tillé 2006, *Sampling Algorithms* (Springer) — PARTIAL | §5.6 conditional Poisson sampling (for the later search phase). Its iterative-cap πps algorithm (section not verified) was this note's FIRST choice and is now **rejected** (review M1): a capped π = 1 on a species not on the true team gives −log 0 in the BCE and a dead gradient. MEASURED: 0.92 % of cold-start pool decisions have a capped species and **0.23 % have one that is not on the team**; 1.7–1.9 % / 0.13–0.27 % under the memorising proxies (`x5_revision_2026-10-03/out/m1m3.log`). |
| Scheirer et al. 2013 (TPAMI); Bendale & Boult 2016, *OpenMax* (CVPR) | Precedent for an explicit "unknown" bucket. Ours comes from a prior plus supervision, not from activations; borrow the idea only. |
| Good 1953 (Biometrika); Ferguson 1973 / Blackwell & MacQueen 1973 | The unseen-mass and "new table" mass literature. It supports OTHER's existence. X5 does not need a Good–Turing estimate, because the unseen COUNT is known (6 − revealed, §8.1 names the < 6-mon exception) and OTHER is the in-model tail of π. |
| Gneiting & Raftery 2007 (JASA 102(477):359–378), §2–3 | **The loss's grounding.** The log score is strictly proper (Example 3); a sum of per-indicator log scores is proper for the vector of MARGINAL presence probabilities by linearity (strictly proper for the marginals, not for the joint). That is the set BCE of §3.2. |
| Guo et al. 2017 (ICML); Gupta & Ramdas 2022 (ICLR; arXiv:2107.08353) | ECE and reliability; Gupta & Ramdas's multiclass-to-binary reduction defines CLASS-WISE calibration, P(Y_l = 1 \| h_l(X) = q) = q per label. **Used for the READ only:** per-species presence calibration (§4, §7.4). Their setting is multiclass and ours is multi-label; the class-wise binary reduction carries over. It does not ground the loss. |
| Brier 1950; Murphy 1973; Spiegelhalter 1986 (Stat. Med.) | Brier score with the reliability / resolution split, and the Spiegelhalter Z for calibration. **Adopt for presence indicators.** X5 should raise RESOLUTION over the blob at equal reliability. |
| Van Calster et al. 2016 (J. Clin. Epidemiol.) | The calibration hierarchy. A count test of Σ predicted against realised is MEAN calibration (level 1). **Adopt for role calibration (§4).** |
| Czado, Gneiting & Held 2009 (Biometrics) — content from memory | The nonrandomized PIT for count forecasts. Used only if a count DISTRIBUTION is read (§4.3). |

**Grounding check on F4, a fixed group mass.** The six team indicators are negatively correlated, because exactly six
are true. So the variance of a realised count is BELOW Σ p(1 − p), and a Z built on Σ p(1 − p) is conservative. §4
uses it knowing that. No reference for dependent-indicator count calibration was found, so the application is ours.

### 2.6 Evaluation methodology (§7) and PPO early stopping (§5)

- **RL evaluation across seeds.**
  - Henderson et al. 2018 (AAAI) and Colas, Sigaud & Oudeyer 2018 (arXiv:1806.08295; power analysis for seeds).
  - Agarwal et al. 2021 (NeurIPS; arXiv:2108.13264), §3.3: stratified percentile-bootstrap CIs that resample RUNS
    within tasks, and the IQM. **It grounds nothing about clustering by battle within a run** (the first version cited
    it for that; corrected).
  - Patterson, Neumann, White & White 2024 (JMLR 25(318)): run-count, CI and multiple-comparison guidance across seeds
    ("in almost all cases 5 runs is insufficient to make strong claims"). **Not** a source for pre-registration (the
    word does not appear; corrected).
  - **Bouthillier et al. 2021** (MLSys): randomise MORE variance sources per run, because seed-pinning one source
    hides variance.
- **Non-inferiority and group-sequential designs.**
  - Piaggio et al. 2012 (JAMA, CONSORT NI); FDA 2016 NI guidance; Schuirmann 1987 (TOST); Wellek 2010.
  - Pocock 1977 (Biometrika 64(2):191–199); O'Brien & Fleming 1979 (Biometrics 35(3):549–556).
  - **Unknown variance:** Pocock's (1977, pp. 195–6) significance-level approach (each look's known-variance nominal
    p-value mapped to a t-quantile on the current df) was shown by exact calculation to be "reasonably accurate"
    (Jennison & Turnbull 1991, Biometrika 78(1):133–141). Jennison & Turnbull 2000 treat group-sequential t-tests
    (§11.5) and error spending, efficacy and futility boundaries (Ch. 7); section numbers via citing papers, the book's
    table of contents was not opened. §7.4 does NOT rely on the approximation: it verifies type-I by simulation and
    re-calibrates the boundary constant.
  - **Non-binding futility:** FDA 2019 adaptive-design guidance, §V.A: "nonbinding futility guidelines … to a trial
    with appropriate group sequential stopping rules for efficacy, does not increase the Type I error probability."
  - **A secondary endpoint in a group-sequential trial:** testing it at full α at the look where the primary crosses
    inflates its type-I error (Hung, Wang & O'Neill 2007, J Biopharm Stat 17(6):1201–1210; analysed by Tamhane,
    Mehta & Liu 2010, Biometrics 66(4):1174–1184; both via a secondary source). §7.4 therefore tests the purpose
    metric against a group-sequential boundary too.
  - Wald 1945 (SPRT). Van den Bergh's Fishtest notes on the pentanomial pair model are grey literature.
  - No methods paper combining group-sequential NON-INFERIORITY, futility and estimated variance was found
    (UNVERIFIED that none exists); the design composes the pieces above and verifies the composition by simulation.
- **Clustered data.** Field & Welsh 2007 (JRSS-B 69(3):369–390): the cluster bootstrap is consistent under the
  transformation and random-effect models. Cameron, Gelbach & Miller 2008 (Rev. Econ. Stat. 90(3):414–427):
  over-rejection with few (5–30) clusters. §4's descriptive CIs resample battles; never with few clusters.
- **Variance reduction.** Deng et al. 2013 (CUPED). Considered and **rejected** for the run-level term, because no
  PRE-treatment covariate exists: both arms differ from step 0.
- **PPO KL early stop (§5, record only).**
  - Schulman 2020, "Approximating KL Divergence" (blog): with x ~ q and r = p(x)/q(x), k1 = −log r and
    k3 = (r − 1) − log r are unbiased estimators of KL[q ‖ p].
  - SB3 PPO: `approx_kl = mean((exp(log_ratio) − 1) − log_ratio)`, log_ratio = new − old, i.e. k3 for
    KL[old ‖ new], checked per MINIBATCH against 1.5 × `target_kl` (code read).
  - Spinning Up PPO: `(logp_old − logp).mean()`, i.e. k1 for the same KL, checked per full-buffer policy step (code
    read).
  - The 1.5 traces to the adaptive-KL rule in Schulman et al. 2017. Andrychowicz et al. 2021 (ICLR) found KL penalties
    redundant under the PPO clip and does NOT study early stopping. **No paper evaluates the early stop itself.**

---

## 3. The design

### 3.1 Token set (production arm `--belief-tokens fixed_mass`)

| group | tokens | mass | revealed | hypotheses | OTHER |
|---|---|---|---|---|---|
| their team | the 6 existing opponent seats + **1 new seat, OTHER_species** | 6 | role token as today, weight 1 | the (6 − r) unrevealed seats hold the top-(6 − r) species by π, each a concrete mon token (§3.4) with presence π_s | Σ_tail π_s, summed directly; masked iff no unrevealed seat (r = 6) |
| their active's moves | the 6 E4 seats; **OTHER_move re-uses the active's E5 tail seat** | 4 | pinned at 1 | the top (6 − r_m) unrevealed moves by π_m | Σ_tail π_m beyond the seats, summed directly; masked iff all 4 moves revealed |

- The sequence goes from 61 to **62 tokens** at the recommended budget (§9 M3 option a adds more).
- The spec said "+2" (§3.9). OTHER_move takes the active's E5 seat because that seat already summarises the mass
  beyond the top-6. In the `fixed_mass` arm its `p_tail` becomes the fixed-mass tail Σ_tail π_m (today it is
  `min(1, Σ tail w)` over sigmoid w, a different quantity). Its `worst_phys` / `worst_spec` features stay and are a
  max-type reduction (§3.5 class M).
- **The bench mons' E5 seats** keep their features (from their own slot's move posterior). Each gets the key bias
  **log π of its owner mon** (0 for a revealed mon). Without it a hypothesis's E5 seat would count as a whole token.
- **Ordering and ties.** One ordering serves every consumer (hypothesis seats, E4 seats, E5 tail membership and the
  op's stash, which today compute top-6 twice under two different tie rules): a STABLE sort by π descending over
  candidates laid out in species (or move) number order, so equal π resolves to the lower number. `torch.topk` does not
  specify tie order and is not used for selection.
- **Rule 8 at the selection boundary.** Which species takes the last seat is a discontinuous function of π. Every
  check that compares two computations of the selection (eager vs compiled, CPU vs CUDA, the golden, the readers)
  EXCLUDES inputs whose k-th and (k + 1)-th π differ by less than 1e-6, and reports the excluded count. MEASURED
  frequency on the pool at cold start: 0–6 decisions per (budget, r) cell of 11,504 decisions; it rises under a
  memorising belief that assigns equal π to co-occurring species (up to ~1,500 in one cell under the exact in-sample
  proxy), which is why the rule is explicit.
- **A second selection boundary (found in U2).** OTHER's embedding averages the NEXT 32 tail candidates (§3.3), so
  the (k + 32)-th / (k + 33)-th π gap is a boundary too, and the move seats have their own. The exclusion applies at
  every one (`hypothesis_set.near_tie_rows`). MEASURED under the Smogon prior (4,000 synthetic reveal sets drawn by
  the prior marginal, r = 1–6): 0 seat-boundary near-ties, 14 tail-cutoff near-ties (0.35 %; species sharing the
  1e-4 floor tie exactly far down the order).
- **Masks are STRUCTURAL** (a count reaches zero), never a threshold on a continuous mass. OTHER's log-mass is computed
  as a logsumexp of the tail's log-presences (§3.2), so it is finite whenever its count is nonzero and needs no floor.

### 3.2 Mass: how π is built and updated within a battle

**Species, at every decision, from the current observation (no recurrent state):**
1. **Scores.** a_s = log P_T0(s | revealed) + δ_θ(s | ctx) over the valid set V = the dex-row table's `valid` nums
   (the 386 base-form species, nums 1–386) minus the revealed nums. V is STRUCTURAL: the sentinel 0, the revealed
   species and the T0 prior's 13 phantom nums 387–399 (floored at 1e-4, no species: F-X5-21) are excluded exactly
   (π = 0, no logit), not by the prior's finite Species-Clause logit. (This line first said "V = {species numbers
   ≥ 1}", which would have carried the phantom nums; corrected in U2.)
   - `P_T0` is the existing Smogon teammate naive Bayes. The prior stays SMOGON-ONLY (owner rule, 2026-08-15).
   - **δ_θ is new and state-dependent:** a Deep-Sets sum-pool (Zaheer et al. 2017) over the REVEALED opponent role
     tokens ⊕ the global token → a 2-layer MLP → species logits. As built (U2): "the global token" is δ_θ's OWN
     linear projection of the TeamTransformer global token's RAW input (`our_ctx_raw ⊕ opp_ctx_raw ⊕
     non_matchup_rest`), because the transformer's `global_proj` is T1; the MLP is LayerNorm(256) → 96 → ReLU →
     400.
   - The last layer is zero-initialised, so at a cold start π is the Smogon prior's fixed-size marginal exactly.
   - It runs in tier T0, on pre-transformer tokens only (`tier_contract`).
   - It replaces the constant `unknown_slot_emb`, so the hidden-slot learned delta finally depends on state (§1.3).
2. **Logistic fixed-size marginals** (LML, Amos et al. 2019; the I-projection reading, Csiszár 1975):
   **π_s = σ(a_s + τ), with τ the root of Σ_{s∈V} σ(a_s + τ) = k, k = 6 − r.**
   - **Bracket (provable).** With n = |V|, a_max and a_min over V: at τ_lo = log(k/(n − k)) − a_max every term is at
     most k/n, so Σ ≤ k; at τ_hi = log(k/(n − k)) − a_min every term is at least k/n, so Σ ≥ k. The width a_max − a_min
     is bounded by the prior's construction (marginal floor 1e-4, lift clipped to ±4 nats, δ_θ's own range):
     MEASURED ≤ 45.2 nats on the pool.
   - **Bisection: 64 fixed iterations**, the same count in every dtype, under `no_grad`, no data-dependent exit, so the
     graph is static and the result deterministic. 64 halvings of ≤ 46 nats reach below the fp64 resolution of τ; fp32
     stops moving at its own resolution after ~25 and stays there.
   - **Tolerance reached** (MEASURED, 11,504 pool decisions × three beliefs, `x5_revision_2026-10-03/out/m1m3.log`):
     max |Σπ − k| = **2.7e-15 in fp64** and **1.5e-6 in fp32**; max |π_fp32 − π_fp64| = 1.5e-7. A Newton step after
     the bisection only lowers the fp32 residual to 1.0e-6 and is not used.
   - **k = 0** (r = 6): π ≡ 0 structurally, no bisection is read, OTHER and every hypothesis seat are masked.
   - **k = n** (every remaining candidate must be present): π ≡ 1 on V structurally, and those indicators are
     EXCLUDED from the BCE (they carry no information). It is unreachable for species (n = 386 − r ≥ 380 against
     k ≤ 6; this line first said "n ≥ 382", corrected in U2) and for moves (a gen-3 learnset's legal moves against
     k_m ≤ 4); a deterministic test asserts the structural branch.
   - **Never 0 or 1 otherwise.** σ of a finite argument; the max π MEASURED is 0.718 under the prior and 0.99999970
     under the exact memorising proxy. The BCE is finite on every decision.
3. **Selection.** The hypotheses are the top-k species by π (§3.1's ordering). **OTHER_species = Σ_tail π_s**, summed
   over the tail directly, and its log-mass is `logsumexp_tail(logsigmoid(a_s + τ))`.

**Moves (the opponent active).**
- The MoveBelief per-move scores (Smogon usage, an INCLUSION rate as Foul Play treats it; §2.4, ⊕ the existing
  learned head) go through the same construction with k_m = 4 − r_m over the 400-move vocabulary minus the revealed
  moves. Revealed moves stay pinned at 1.
- Hidden Power keeps its typed composition (`compose_typed_hp`) before the construction.
- The E4 seats are the top (6 − r_m) unrevealed moves by π_m; OTHER_move takes the rest (§3.1).
- A hypothesis seat's own move posterior (from its species-specific token, §3.4) uses the same construction at k = 4.

**Gradient path (orchestrator decision, review M10).**
- **π is DETACHED where it enters the attention biases, OTHER's mass, the flat pointer's presence weighting and every
  op reduction.** δ_θ is trained ONLY by the presence BCE. The policy still learns from token CONTENT, and attention
  can still down-weight a key through its query-key logit.
- **τ needs no gradient.** ∂BCE/∂τ = Σ_s (π_s − y_s) = k − k = 0 whenever the labels count exactly k true unseen
  species, which they do by construction (F-X5-3's guard makes a missing label a FAULT). So computing τ under
  `no_grad` gives the exact implicit gradient; LML's implicit backward is not needed.
- **Why detach.** Undetached, RL could tune π as a GATE (push a hypothesis to π ≈ 0 to silence a token, or inflate
  OTHER) with no regard for truth. π would then stop being a calibrated probability, and the purpose metrics (presence
  Brier, OTHER and role calibration) would measure an RL-tuned gate instead of the supervised belief.
- **What is lost.** The return cannot shape which species the belief ranks high, so a belief error that matters to
  play is corrected only as fast as the BCE corrects it. **How the A/B would show it:** it cannot isolate it. A
  NON-INFERIOR verdict says detaching cost less than δ; a NOT DETECTED one is followed by the cheap ablation in §8.1.
  Undetaching is a separate lever, never folded in.

**Within-battle updating.**
- Every reveal changes r (or r_m) and the conditioning set, and π is recomputed at the next decision: the same
  "re-condition on each reveal" as `design_q_head.md` §1.
- Not modelled: NEGATIVE evidence, such as "they did not switch to X when it was the obvious answer". That needs the
  learned roster mixtures (X12) and is recorded as a gap (§8.2).

**Supervision.**
- **Set BCE** on π against the true unseen set, one binary indicator per valid species, logits a_s + τ exactly (no
  clamp). A sum of per-indicator log scores is proper for the marginal presences (Gneiting & Raftery 2007, §2.5).
- It needs NO slot matching. Unseen slots are exchangeable in gen 3 (no team preview), so the Hungarian step goes.
- **OTHER's calibration then follows by linearity:** E[number of true unseen species outside the list] = Σ_tail π
  when π is calibrated.
- Labels: the existing `belief_species [6]` with the revealed mask. **No new Rust label.**
- **BeliefHead is kept as the post-transformer trunk-shaping aux**, re-targeted to the same set BCE, so the arm
  changes the representation and not the shaping signal (one lever).
- The coefficient stays at 0.05; `belief_grad_mode` stays `shaping`.

### 3.3 What OTHER is, and how much it carries (review M3)

- OTHER is an ENTITY, not a scalar.
- **Its embedding** is a learned "rest" vector plus a linear map of the π-weighted mean SPECIES embedding over the
  next 32 tail candidates (the move token likewise).
- It is a key in attention with its log-mass bias, a pointer candidate, and (when X4 returns) a Q column.
- **Its PHYSICS is an owner choice (§9 M3).** The first version gave it none, to avoid "rebuilding the blob". The
  review showed that this leaves most of the hidden mass unpriced, and the numbers below confirm it.

**How much mass OTHER carries** (MEASURED; `x5_revision_2026-10-03/out/m1m3.log`; 719 pool teams × random reveal
orders; r = 0 is omitted because the lead is always revealed). OTHER's share = OTHER mass / k; recall = the share of
the true hidden mons that are in the hypothesis list.

| belief | budget H | r = 1 | r = 2 | r = 3 | r = 4 | r = 5 |
|---|---|---|---|---|---|---|
| Smogon prior (cold start) | k (1:1 seats) | 66 % · 0.42 | 64 % · 0.43 | 64 % · 0.42 | 67 % · 0.40 | 74 % · 0.34 |
| | k + 2 | 56 % · 0.52 | 53 % · 0.55 | 50 % · 0.56 | 49 % · 0.57 | 49 % · 0.58 |
| | k + 6 | 41 % · 0.69 | 36 % · 0.73 | 31 % · 0.74 | 28 % · 0.76 | 25 % · 0.77 |
| | 12 | 38 % · 0.73 | 29 % · 0.78 | 22 % · 0.82 | 16 % · 0.85 | 12 % · 0.88 |
| | 18 | 24 % · 0.87 | 17 % · 0.89 | 12 % · 0.91 | 8 % · 0.94 | 6 % · 0.94 |
| on-pool proxy (pool naive Bayes, in-sample, fitted per r to the head's NLL) | k | 61 % · 0.47 | 60 % · 0.49 | 62 % · 0.45 | 66 % · 0.41 | 74 % · 0.34 |
| | k + 6 | 32 % · 0.74 | 27 % · 0.80 | 26 % · 0.79 | 25 % · 0.77 | 25 % · 0.77 |
| | 12 | 28 % · 0.78 | 20 % · 0.85 | 16 % · 0.87 | 14 % · 0.87 | 12 % · 0.88 |
| on-pool proxy (EXACT in-sample team memory, fitted) | k | 61 % · 0.47 | 57 % · 0.52 | 64 % · 0.45 | 67 % · 0.40 | 74 % · 0.34 |
| | k + 6 | 32 % · 0.74 | 23 % · 0.83 | 31 % · 0.80 | 28 % · 0.77 | 25 % · 0.77 |

How to read it:
- **The on-pool sharpness proxy is a FLOOR on sharpness, not an estimate.** The banked head (L21301) reaches a per-slot
  matched NLL of 2.24 / 2.51 at k = 1 / 2 reveals. No team-level belief built from SPECIES alone reaches that, not even
  exact in-sample memorisation of the 719 teams (3.02 / 2.64). The head also reads revealed moves and items, which can
  identify a pool team outright. Running it read-only needs its own pin and battles (`belief_calibration` §1), so it
  was not cheap; the live head's OTHER mass is UNVERIFIED and could be much smaller on-pool.
- **Calibration of the mass itself** (prior, budget k): mean OTHER mass 3.32 / 2.58 / 1.94 / 1.34 / 0.74 against a
  realised out-of-list count of 2.91 / 2.29 / 1.75 / 1.20 / 0.66. The ladder-fitted prior over-states OTHER on the pool
  by about 12 %, as expected of a prior that is not pool-memorising.
- **Is no-physics OTHER a regression against today's blob? YES, on the physics channel.** Today's averaged defender
  prices 100 % of the hidden mass (with a Jensen gap). With OTHER at zero physics and a 1:1 budget, X5 prices only
  26–43 % of it concretely and none of the rest: 57–74 % of the hidden mass would read as absent to the op. §3.7 shows
  the cells would then read OTHER as IMMUNE (`e_mult` = 0) and drop its share of `e_high` / `e_pko`. That would be a
  silent regression riding on the A/B. §9 M3 lists the options; the recommendation (c) closes it at no token cost.

### 3.4 How hypothesis tokens are built, and the physics they get

- **Content.**
  - A hypothesis seat is a CONCRETE mon of species s, encoded by the SAME `pokemon_encoder` as a revealed one.
  - Its per-mon observation row comes from a static **dex-row table [max_species = 400, 122]**, one row per
    BASE-FORM species at its national-dex num (386 rows, nums 1–386; the rest zero, with a `valid` mask),
    produced by THE observation encoder: the Rust slot writer `BattleVersion::encode` calls per mon, fed the one
    synthetic input "species s present, unrevealed set, full HP, no status" (`encoder::hypothesis::hypothesis_slot`,
    BUILT in U1; `designs/rust_sim/encoder.md` §10).
  - An unseen gen-3 mon is pristine: it has never been on the field, so HP, status and boosts are exactly the defaults.
  - The table is a **committed artifact beside the model code** (`src/agents/model/hypothesis_dex_rows.json`, loaded by
    `agents.model.hypothesis_dex_rows.load_hypothesis_dex_rows`), not under `data/`, so a pinned run isolates it.
  - **Two gates (U1, BUILT).** (i) A `sim`-tier gate regenerates the table with the encoder and requires byte equality.
    (ii) A **real-state cross-check**: over a fixed seeded set of real Rust battles, at each opponent mon's first
    appearance, its real encoded row must equal its dex row byte for byte on every cell outside a DECLARED list
    (`encoder::hypothesis::CELLS`). As built, the list is the mon's ON-FIELD state — status, HP fraction, the status
    counters, the sleep belief, recency, last action and the active flag — plus an item, ability or move block the
    FIELD revealed before that decision, read from the real row's own reveal flag (Leftovers at the end of the entry
    turn, an ability announced on entry). Any other differing cell FAILS, naming it. The first version listed only
    "the active flag, an ability announced on entry, the field position"; the slot has no field-position cell, and the
    other on-field blocks differ legitimately at a real first appearance (changed in U1, see the Decision record).
    This catches a synthetic state that the encoder renders differently from a real one.
  - No second encoder exists, and the runtime observation does not change.
  - A learned `hypothesis_marker` vector is added to the token (as E5's `tail_marker` is). The token-type table is not
    grown, because that would change every state_dict.
  - **As built (`gen3_x5_hyp_gather_v1`, `agents/model/hypothesis_encode.py`, 2026-10-05): the encoding is computed
    as the per-row pass's EXACT split, not by re-running the encoder per row.** The per-row pass (`PokemonEncoder` on
    `hypothesis_ctx`, all 12 mons of every row) stays the definition. The encoding is NOT a function of the species
    alone: five row-level inputs reach it — the clock, weather, fainted and hazard features (the move network's
    context) and the screens (with the first four, the role encoder's broadcast global context) — plus the active-
    context scatter, which a hypothesis never receives (it is never the active; 0 of 752 hypothesis slots on 1,024
    real rows). MEASURED on those rows: one species' tokens differ across rows by up to 1.06; with the row-level
    inputs held at one row's values they are bitwise equal per species (`measurements/x5_hyp_gather_2026-10-05/`).
    Each row-level input enters only through the FIRST Linear of the move network and of the role encoder, which is
    additive over its input columns, so the split is exact: the species-only columns go through each first Linear
    ONCE per forward over the 400-row dex table and are gathered by hypothesis species (when B·6 < 400 — T2's
    buckets 8 and 64 — over the slots' own dex rows instead, a static choice); the row-level columns once per row;
    everything after the first Linears per OPPONENT slot only (6, not 12). Equal to the per-row pass up to fp32
    reassociation: |Δ| ≤ 2.9e-6 on token values up to 5.8 (fp32), ≤ 6.2e-15 (fp64); the gradients agree at fp64.
- **The aux heads on hypothesis seats.** MoveBelief, ItemBelief, HP-type and spread read a SPECIES-SPECIFIC token for
  hypotheses, through the existing per-species Smogon priors, so the hidden slot stops being a constant. Supervision:
  - **Moves: supervised iff the hypothesis species IS on the true unseen team**, against that mon's true moveset
    (`belief_moves`, which exists for unrevealed mons: `belief.rs:266-283`). The head then models "the set GIVEN the
    species is present", an exact label; a hypothesis not on the team is masked. Structural, no threshold.
  - **Item, HP type, spread: unsupervised on hypothesis seats.** Their labels exist only for revealed mons
    (`per_slot.rs:340-348`, `spread.rs:185-199`). A new label family would be needed; out of scope (F-X5-16).
- **Physics.**
  - The damage op prices each hypothesis seat as a concrete DEFENDER: its own stats, no averaging. It also prices it
    as an ATTACKER: `att_gate` opens for hypotheses, using its per-species move posterior.
  - **"Alive" is rebuilt from `opp_addressable`,** not from `hp > 0`. A hidden slot encodes HP exactly 0 today, which
    silently drops it from `p_pur_vs_us`, the v / t / g edges, the speed gate and s1 (F-X5-12). A hypothesis row is at
    full HP from the dex table, but the gate must not depend on that.
  - Each op row is then weighted by presence wherever it is reduced (§3.5). Effect: E[f(x)] replaces f(E[x]) for the
    hypotheses.
  - OTHER's physics: §9 M3.
  - **As built (U3 part 3, `hypothesis_tokens.OpRoster` on `op.stash.x5`).** The op runs on the HYPOTHESIS context, so a
    hidden slot is priced as its species at first appearance (full HP, no status, the dex row's ability prior). The
    DEFENDER is the expected-latent read on a per-slot one-hot of the species (its own E[mult], E[def / spd / maxhp]).
    **P(KO) is UN-nulled for hypotheses** (ORCHESTRATOR brief: decide and justify): the blob nulls it because the
    AVERAGED defender's KO is a threshold of averaged stats, where the Jensen gap is worst; a hypothesis is one
    pristine species, so P(KO | s, full HP, neutral spread) is a defined, species-exact quantity — and an OHKO on a frail
    or 4×-weak switch-in is exactly the plan-relevant fact X5 exists to add. OTHER's averaged defender KEEPS the null.
    "Alive" is `opp_addressable` in every opponent-slot gate. Every live opponent mon ATTACKS (C1b / C2 / C3 / D4) with
    its own species' stats and STAB and its own fixed-mass move presence (k = 4 − revealed over its legal moves,
    revealed pinned at 1; the active's row is the move group's `w_all`), its candidates the first K of one stable
    per-mon order. A per-(seat, mon) cell is NOT scaled by the mon's π: it is "what this mon does if present", and its
    presence is the trunk's log-π key bias — scaling both would count presence twice. The one max over MONS
    (`p_pur_vs_us`) is presence-scaled (slot π; OTHER with `other_any`).
- **Rejected:** a torch-side builder of synthetic observation rows. That is a second encoder and a drift risk. The
  synthetic-key registry exists because two encoders of one row drifted before.

### 3.5 How tokens enter the phase chain, and what each reduction means (review M2)

| tier | change |
|---|---|
| T0 RESOLVE | δ_θ, the fixed-size construction, hypothesis selection, dex-row gather, `pokemon_encoder` on hypothesis rows, OTHER embeddings. `BeliefSlots` is retired in this arm. |
| T1 REASON | The op prices hypotheses (defender + attacker). Edge cells on hypothesis seats as on revealed ones. The `TeamTransformer` bias adds **log π_j on every query's logit for key j, in all heads**, for opponent seats, OTHER, the E4 / OTHER_move seats and the bench E5 seats (owner's π). Revealed keys get 0, masked keys −1e9 (the key-padding addend that already exists). |
| T2 DECIDE | `their_cls`, `value_cls`, `HiddenOppBeliefPool` and `value_entity_pool` get the SAME log-π key bias. The three `nn.MultiheadAttention` / decoder pools take BOOL key-padding masks today; they need FLOAT masks (F-X5-14; behaviour under compile UNVERIFIED, U3 checks it first). The flat α pointer (§3.7) replaces `alpha_head` + `beta_head`. The intent, pair, switch-branch and conditional-threat cells read α through §3.7's re-expression. |
| T3 DELIVER | Unchanged. |

**Every reduction over the opponent-mon or opponent-move axis, and its declared presence semantics.** A read-only code
survey (2026-10-03) found **41** in the production forward, by grepping every reduction primitive (`amax`, `max(`,
`topk`, `logsumexp`, `softmax`, `mean(`, `sum(`, `prod`, `argmax`, `einsum`, `@`, key-padding masks) over every
non-test `src/agents/model/*.py` and reading every forward module. The damage-op files were checked line by line; the
consumer cells were traced more lightly, so U3 re-runs the census against the built code before any test is written.

| class | sites (file:line, under `src/agents/model/`) | presence semantics | tests |
|---|---|---|---|
| **E: expectation-type.** Softmax attention and pools; α / β-weighted sums | TeamTransformer (`team_transformer.py:56`, bias 346-365); `their_cls` (`pools.py:79`); `value_cls` (`:111`); HiddenOppBeliefPool (`:167`); `value_entity_pool` (`value_readouts.py:104-107`); α-weighted sums in `pair_outcome.py:315-322, 345, 395, 454-455`, `conditional_threat.py:182-185`, `intent_threshold.py:132-135`, `intent_move_cell.py:93-99`, `intent_conditional.py:169-193`, `switch_branch.py:248-269`; `pair_reduce.py` R1's E[cell], E[cell²]; MoveBelief reinjection (`belief_heads.py:402`); `opp_p_ghost` (`damage_op.py:958-969`) | log-π key bias, or π-weighted sums over the flat pointer's events | **I1 and I2, exact in fp64, to 1e-6 in fp32**, per site |
| **M: max-type** | the op's 8 incoming `(w·value).amax` (`damage_op.py:836-841`); Choice Band (`:845-846`); argmax accuracy / provenance (`:858-866`); c1b, c2, c3, d4 amaxes (`damage_op_pairwise.py:397-417, 586-587, 664, 789, 846-847`); `p_pur_vs_us` over opponent mons (`:980`); E5 `worst_phys` / `worst_spec` (`pointer_head.py:151-154`); R1 `prov = Σα·w` = Σw²/Σw (`pair_reduce.py:150-154`) | **an owner choice, §9 M2.** Recommended: today's presence-scaled max, max_m (π_m · v_m), with I2 declared FALSE | I1 exact (values ≥ 0 and π multiplies before the max); a regression test that the output on revealed-only inputs equals today's formula; a determinism test with §3.1's near-tie exclusion |
| **S: selection** | top-K for the op stash and E4 (`damage_op_blocks.py:875, 961`); per-mon `_believed_attackers` (`damage_op_pairwise.py:275`); E5 tail membership (`pointer_head.py:147-150`) | rank-based, discrete; ONE ordering (§3.1) | tie-rule determinism with the near-tie exclusion |
| **O: other** | Sleep Clause OR (`damage_op_blocks.py:626, 1163`); `has_opp` `.any()`; `slot_live` (`:920-922`); P_T0's revealed evidence (`t0_species.py:83-85`) | independent of hypotheses | none new |
| aux only | the β label's max (`opp_intent.py:194`, retired with β); `belief_bank` matching (re-targeted, §3.2); B centring (`ridealong_heads.py:784-788`) | — | — |

**The census RE-RUN on the built code (U3, 2026-10-04, read-only, at `7b46d62f`).** No site in the table above has
disappeared; the line numbers moved (e.g. the op's incoming amaxes now `damage_op.py:862-865`, the op top-K
`damage_op_blocks.py:905`, `refine_candidates` `:991`, `p_pur_vs_us` `damage_op_pairwise.py:1057`). Four sites were
MISSED: Beat Up's opponent-party sum `damage_kinds.beatup_party_opp` (class E, linear — added after the first census),
`pair_outcome.pair_alpha_full` (class E, a distinct α split), and the α / β pointer heads (`opp_intent.py:81-126`;
retired by §3.7, U4). Two were MISCLASSIFIED: d4's per-mon `topk` (`damage_op_pairwise.py:851`) is class S, not M; E5's
`p_tail` (`pointer_head.py:150`) is a linear Σ (class E), not S. The census also confirmed that every hidden bench mon
is multiplied by 0 as an ATTACKER today (`att_gate`, d4's gate, `hp > 0` — the "only Beat Up's party sum and its E5
seat" exceptions), and listed every opponent-slot `hp > 0` gate F-X5-12 named plus x's `opp_cells` (`:1068`).

**As built (U3 parts 1–2, `gen3_x5_belief_tokens_v1`).**
- *Class E, opponent mons — BUILT:* the TeamTransformer (log π on their 6 slots, the E5 bench seats by owner, OTHER,
  and — part 2 — log π_m on the E4 seats, OTHER_move's log-mass on the active's E5 seat), `their_cls`, `value_cls`,
  `HiddenOppBeliefPool`, `value_entity_pool` (OTHER joins each; FLOAT key masks in the three `nn` pools, `−inf` on a
  masked key). F-X5-14 checked FIRST on CPU Inductor: the three pools compile and match eager to ≤ 3.8e-6, and the
  WHOLE fixed_mass extractor compiles (100 s cold) and matches eager to 2.7e-6 (pi) / 1.3e-6 (vf) on 64 golden rows
  (0 near-tie rows excluded). GPU: DEFERRED (U8). NOT yet: Beat Up's party sum, `opp_p_ghost` (part 3); the α / β
  consumers (U4). **Part 3:** Beat Up's party sum (π / k over every candidate) is BUILT; `opp_p_ghost` follows the
  per-slot one-hots (the hypothesis's own Ghost bit).
- *Class S, the active's moves — BUILT:* ONE order for the E4 seats, the op's seat axis, α's seats, D3 / S3 and the
  intent operands (no `torch.topk`; F-X5-13 closed for the active). NOT yet: the per-mon `_believed_attackers` /
  d4 top-Ks and the bench E5 seats (still sigmoid top-K, as the blob). **Part 3: BUILT** — one stable per-mon order
  over each mon's fixed-mass move presence feeds `_believed_attackers` (C1b / C2 / C3) and D4; the bench E5 seats'
  tail is the moves beyond rank K of it; the cuts join `near_tie_rows` (`slot_moves_tie_gap`).
- *Class M, the active's moves — BUILT (§9 M2 = C):* the op's 8 incoming amaxes, Choice Band and the argmax
  accuracy / provenance weight each candidate by its fixed-mass presence (π_m; 1 revealed); OTHER_move's
  `worst_phys` / `worst_spec` are presence-scaled maxes over its members. I1 exact, I2 DECLARED FALSE and pinned.
  **Part 3: BUILT** — the c1b / c2 / c3 / d4 maxes run over each mon's candidates at their per-mon fixed-mass
  presence; the one max over the MONS, `p_pur_vs_us`, weights each slot by its π and takes OTHER with `other_any` =
  1 − Π_tail(1 − π) (ORCHESTRATOR F4 (b)).
- *OTHER's physics (M3 (c), F4 (a)) — BUILT in part 3:* `other_roster` (an OTHER-mode pass of the op's kernels, every
  hidden slot holding the tail-averaged mon) gives OTHER's column for D1, C1, C3, D4 and V, written to its trunk seat's
  edges (F-X5-28 closed). C2 / S1 / T / X / G carry no OTHER edge (F-X5-29).
- *The revealed Hidden Power seat:* priced as E_t[f(HP_t)] through an extended seat axis (the K seats ⊕ the 16 typed
  nums) contracted by `mix` — never as the BP-0 typeless 237.

**The two invariances, scoped to class E.**
- **(I1) Zero = mask:** a hypothesis with π = 0 gives the same output as that seat masked. A presence of exactly 0 is
  the masked case, so there is no boundary.
- **(I2) Split = whole:** replacing one hypothesis (w) by two identical tokens (w/2, w/2) leaves every attention
  output unchanged (the ToMe identity) — the copies' total MASS equals the whole's. As tested (U3): a hypothesis slot
  holding OTHER's token at w/2 plus OTHER at w/2 equals OTHER at w with that slot masked, through each site's real
  forward API (1e-12 fp64, 1e-5 fp32; I1 bit-exact in both). For the **flat pointer**, I2 reads: the probability of each EVENT (the sum
  over its copies) and of every other candidate is unchanged; the per-candidate output vector changes shape. As
  tested (U4, `flat_intent_test::test_I2_for_the_pointer_the_mass_of_copies`): a move seat of presence w split into
  two copies of w/2 through the head's real forward (one more seat) gives P(event) = the copies' sum and every other
  candidate's probability unchanged, 1e-12 in fp64; I1 (π = 0 equals the candidate masked) likewise.
- A class-E reduction that fails either test is a bug: a token that "counts as a whole mon".
- **Why class M cannot have both.** For a max, I1 needs π to scale the value before the max; I2 then fails, because
  max(w/2 · v, w/2 · v) = w/2 · v ≠ w · v. No max-type rule is both a function of presence-weighted copies and
  invariant to splitting them. The first version's claim that every reduction would pass both tests was wrong.
- **Each species and each move appears once in the candidate list**, so a real split never occurs. I2 is a test of
  the reduction's MEANING, not a runtime event, and class M's meaning is declared instead (§9 M2).

**What the heads read.**
- The policy pointer (our actions) and the critic read the same refined tokens and pools as today, now weighted.
- The ride-along A head is unchanged: it reads our per-action tokens.

### 3.6 Capacity and compute

**Base figures (MEASURED; production policy instantiated on CPU, `build_learner(production_args())`).**
- 3,065,882 parameters.
- 56.65 MFLOP per row forward (matmul + attention counted), of which the TeamTransformer is 63 %.
- `train_ms` 41.03 s [40.77, 41.18] at N = 256, E10 (sizing B).
- T2 flush 13.12 ms (trainee 4.48 ms; opponents dispatch-bound).
- Memory 7,798 MiB with 2,218 MiB headroom on the 12 GiB card.

| item | Δ | tag |
|---|---|---|
| parameters | δ_θ ≈ +60k, OTHER embeddings ≈ +7k, marker +128, `BeliefSlots` −768, α+β (65.9k) → flat pointer (≈ 45k); net ≈ +45k (+1.5 %) | ESTIMATED |
| forward matmul FLOPs | +1 token: ≈ +0.65 MFLOP/row (+1.2 %), TeamTransformer +1.8 %; hypothesis gather replaces the `P_T0 @ table` matmuls (≈ 0) | ESTIMATED from MEASURED base + analytic per-token cost |
| the fixed-size construction | 64 bisection steps × [B, 400] sigmoid-and-sum for species, × [B, 400] for moves, no grad: ≈ 0.05 MFLOP/row of elementwise work. **Compile unrolls the 64 steps; MEASURED in U2 (CPU, Inductor, cold caches, B = 256, 4 threads): one construction compiles in ≈ 9.0 s against ≈ 1.0 s with zero steps (≈ 1.9 s at 8 steps, 3.7 s at 32 — superlinear in the step count); the whole builder (species + moves) ≈ 18 s beyond the Inductor warm-up, and runs 4.6 ms compiled vs 12.6 ms eager per call.** The learner's micro-step adds a third construction (BeliefHead's set BCE). GPU compile: MEASURED 2026-10-04, F-X5-22 (the 64 steps cost ≈ 0 on the GPU; the CPU figure does not carry over) | ESTIMATED (FLOPs); MEASURED (CPU and GPU compile) |
| op elementwise | hypothesis defenders replace averaged ones (the same `[B,4,6]` shape); the **attacker gate opening adds per-hypothesis attacker rows**; M3 option (c) adds one averaged OTHER row, the cost the blob pays today | **NOT MEASURED** (the FLOP counter ignores elementwise work) |
| M3 option (a), + m hypothesis tokens | ≈ +0.65 MFLOP/row per token: m = 6 is ≈ +7 % forward, plus op rows ∝ the budget | ESTIMATED |
| `train_ms` | +0.3 to +0.6 s (+0.7–1.5 %) from the token; the op term unknown | ESTIMATED |
| T2 flush | ≤ +0.5 % (trainee only; opponents dispatch-bound) | ESTIMATED |
| memory | saved activations ≈ +1 %, ≈ +30 MB at micro-batch 2,048, against 2,218 MiB headroom | ESTIMATED |

**Pre-registered cost budget (build unit U8, GPU via `scripts/ops/gpu_lock.sh`, the learner benchmark).**
- **X5 `train_ms` ≤ +5 %, T2 flush ≤ +3 %, steady-state D-6 headroom ≥ 1,024 MiB at N = 256 with the X26 heads.**
- Over budget ⇒ STOP and report before any A/B GPU. The measured slowdown also enters §7's wall-clock rule.
- **Early read, MEASURED 2026-10-04 at `e78884c4` (U3 part 1 only; two launches per arm, quiet updates; `designs/research_state/measurements/x5_u2_gpu_checks_2026-10-04/README.md`): `train_ms` 43.44 s vs 39.97 s = +8.7 % (over the +5 % line); trainee T2 GPU wait +11.7 % (4.32 → 4.84 ms at B = 256, regime B); `UpdateFit` headroom 2,143 → 1,963 MiB (−180; floor 1,024) WITHOUT the X26 heads (OFF in `--arch production` here, so the "with the X26 heads" figure is UNVERIFIED).** The extractor's compiled fwd+bwd rises 71.7 → 83.1 ms (GEMM +7.4, Triton +3.6, attention ±0; the bisection ≈ 0.4); the real U8 number comes after U3 parts 2–3 and the op.
- **What U3 part 3 ADDS to that cost (not measured on the GPU, not optimised — the cost / ablation pass after U3
  owns it; F-X5-30).** (1) The OTHER-MODE pass re-runs whole six-column kernels to read OTHER's one column: D1, C1's
  five outgoing worlds + C1b, C3, D4 and V — roughly a second copy of those edge families' elementwise work.
  (2) Two more fixed-size constructions per forward, both `no_grad`: the per-mon move presence over `[B, 6, 400]` (six
  rows per decision, 64 bisection steps) and OTHER's averaged moves over `[B, 400]`, plus a stable `argsort` over
  `[B, 6, 400]` for the per-mon order. (3) The op on the hypothesis context, every live mon an attacker in C1b / C2 /
  C3 / D4 (the same tensor shapes as the blob, whose hidden columns were computed then zeroed) — no new shape, but no
  longer zeroed work. (4) OTHER's edge writes: one `Linear(cell → 2·heads)` per OTHER family over `[B, 4 or 6, 1]`
  (≈ 0). MEASURED on CPU only: the fixed_mass extractor's cold Inductor compile rose 100 s (part 1) → 142 s (part 3).
- **What U4 ADDS (not measured on the GPU, not optimised — U8 / the cost pass own it).** (1) OTHER_move's seat-axis
  column prices every per-candidate quantity on the FULL move axis and contracts it with the tail weights: the eight
  status / tempo coordinates (`pair_outcome_coords`) over `[B, 6, M = 400]` (the K-seat path prices `[B, 6, K + 16]`),
  a second `_damage_rolls` pair over `[B, M]` for the c2 operands (our active as the defender), a `[B, M] @ [M, T]`
  type contraction, and `[B, 6, M, 6]`-sized einsums over the damage cells the op already holds. (2) The flat head:
  one shared 2-layer scorer over K + 8 = 14 candidates (24,833 parameters at width 64, against α + β's 65,923, which
  are retired). (3) When the edge families do not include `d1`, one more OTHER-mode outgoing pass for `out_cells`
  (production includes `d1`, so 0 there). MEASURED on CPU: the fixed_mass extractor's cold Inductor compile 145 s
  (part 3: 142 s; one run each, not a controlled A/B) and compiled = eager to 2.4e-6 (pi) / 1.3e-6 (vf) / 6.3e-7 (the
  flat logits; identical −inf pattern) on 64 golden rows, 0 near-tie rows. F-X5-37.
- **U8 cost read, MEASURED 2026-10-04 at `889add9d` (U3 done), WITH the X26 heads, quiet updates (`designs/research_state/measurements/x5_cost_ablation_2026-10-04/README.md`): ALL THREE LINES FAIL — `train_ms` 40.96 → 47.80 s = **+16.7 %** (regime B; the update is regime-independent on blob); T2 flush + GPU wait per host step 21.8 → 29.6 ms = **+36.0 %** (regime A, a seeded 20-snapshot pool); `UpdateFit` headroom 2,074 → **432 MiB** (regime A; the launch is REFUSED below 1,024). Ablation of the extractor's compiled fwd+bwd at B = 2,048 (+16.5 ms): the hypothesis `PokemonEncoder` pass 58.6 %, an unattributed residual 22.7 %, the bisection 5.1 %, the argsorts 4.8 %, the OTHER pass 4.1 % (but ≈ 0.2 ms of the 0.6 ms per T2 forward at B = 8), the K+16 axis 3.5 %, the per-mon construction, attacker gating and key bias ≈ 0. A fixed_mass production launch also hits four refusals today (D-6, a 1.1 MiB T2 slot-load allocation, a compile-sentinel cosine 0 on one launch of four, K9(b)'s tie share 0.54–0.58 vs 0.15): the README's §1.** Ranked optimisation proposals there; the design is unchanged. NOT measured: U4 (`350fb83b`, the flat pointer + OTHER_move's full-axis pricing above) and U6's F-X5-44 fix (`61be9ec9`, a dead OTHER's ties — plausibly the K9(b) tie share; UNVERIFIED for regime A), both landed after `889add9d`.
  - **F-XC-4 status (the compile gate, 2026-10-05): FIXED — a fixed_mass `--compile-trainer` launch passes its startup gate on CUDA (`gen3_fm_index_max_v1`, `measurements/x5_fxc4_nanfix_2026-10-05/`; the gate fixes `gen3_gate_nonfinite_named_v1` + `gen3_gate_independent_arms_v1`, `measurements/x5_fxc4_compile_gate_2026-10-05/`).** The gate's two sides were independent and R1 IS one compiled graph under fixed_mass, so X5's cost is a compiled cost. The "cosine 0.000000" was a NON-FINITE compiled backward on CUDA (41 parameters NaN, eager finite, 8 of 8 compiles at `84f8569f`) that `_cos` read as orthogonal. MECHANISM (a per-kernel NaN scan of the generated code): the first NaN-writing kernel is the backward of the incoming direction's ten channel maxima over fixed_mass's FULL 400-wide candidate sweep (blob prices the top-6). `amax`'s backward is `grad · (x == amax) / Σ(x == amax)`; Inductor recomputed `x` in that kernel and Triton contracted its `a·b + c` into FMAs differently from the forward kernel, so on some rows no element equalled the saved max and the gradient was 0/0. With Triton's FMA contraction off the same graph is finite; `remove_noop_ops` only moved which nodes were recomputed. FIX: under fixed_mass those ten maxima are selected BY INDEX (`damage_op.max_by_index`, a gather at the detached argmax: the value bit-identical to `amax`, the backward a scatter at a saved index). Blob keeps `amax`; its R1 and T2 compiled code is byte-identical before and after (§7.5 holds). The eager change is a tie rule (the gradient goes whole to the first maximal element); the fixed_mass K9 golden did not move. The class stays LATENT at every other `amax` whose input a compiled backward recomputes (blob's included); the gate's `NonFiniteGateArmError` and the canary catch it.
  - **F-XC-5 status (K9(b)'s tie share, 2026-10-05, `gen3_behaviour_tie_identity_v1`, `measurements/k9_tie_identity_2026-10-05/`): the CHECK is fixed, and regime A on the GPU now passes it — 0 at update 1 (selection-free), 0.090–0.113 on updates 2–5 against the 0.15 ceiling, max |Δ log π| exactly 0 (`measurements/x5_fxc4_compile_gate_2026-10-05/` "F-XC-5"; DEVIATIONS: the learner eager, `--no-compile-trainer`, since the compile gate refuses fixed_mass at HEAD; a fresh trainee against a pool built from a one-update checkpoint; 5 updates). That run also found a blocker, FIXED 2026-10-05: a fixed_mass self-play run crashed at its first logger dump once pool rows existed (the stdout table's 36-character truncation made `flat_switch_target_recall_top1_bot` and `_pool` one key; the key is now `flat_switch_tgt_top1`, F-X5-48).** On a CPU Rust-collector rollout at HEAD the fixed_mass arm excluded 13.7 % of rows at the fresh init and 6.3 % at perturbed weights under the old rule. That does not reproduce regime A's 0.54–0.58 at `889add9d`. Under the new rule the figures are 0 % (a fresh run's first update is SELECTION-FREE: the zero-init action scorers mean no tie can move log π) and 5.1 % (the dominant-move argmax's payload-identical ties are cleared). The dry-update fixture's 0.45–0.47 was F-X5-44's, and that fixture is now selection-free too. Of the remaining fixed_mass exclusions, 94 of 2,048 perturbed rows are the stable order's exact π ties, which flips show cannot move log π. They stay excluded because no declared rule proves them; a slot-order-invariance rule would claim them. **Since `gen3_behaviour_tie_consumed_v1` (2026-10-06, `measurements/k9_early_probe_2026-10-06/`)** each caller of the one sort declares how it reads the order: the op's per-mon move orders are a SET before each cut (a random permutation of every per-mon top six leaves log π bit-identical), the species order is read in order up to k, the move group in order. `rb_x5ab_fm_s1006` had stopped at its update 1 on the ceiling (0.281; the other seeds 0.04–0.09) on ties INSIDE the per-mon top six. At its dumped weights on CPU the share goes 15.2 % → 5.9 %, perturbed 5.1 % → 3.8 %, and no row a flip proves distinct is cleared.

  - **The hypothesis encoding split and gathered (`gen3_x5_hyp_gather_v1`, 2026-10-05,
    `measurements/x5_hyp_gather_2026-10-05/`; §3.4 "As built"). MEASURED with the X26 heads, regime A, desktop
    stopped, quiet updates: `train_ms` 49.16 s at `c7b4d03e` → 46.39 s, against blob 41.12 s (same day): +19.5 % →
    **+12.8 % (OVER the +5 % line)**; T2 flush + GPU wait per host step 31.80 → 31.68 ms against blob's 21.76 ms
    (carried from `889add9d`, its T2 code byte-identical since): +46.1 % → **+45.6 % (OVER the +3 % line)**; `UpdateFit`
    headroom 1,106 → **1,192 MiB (PASSES the 1,024 floor; 168 MiB margin)**.** The encoding's own cost on the extractor
    harness (compiled fwd + bwd, B = 2,048) falls 10.09 → 4.38 ms per micro-batch (−57 %), not to ≈ 0: a hypothesis
    token is not a function of its species (the clock / weather / fainted / hazard / screen features enter it), so
    only the two first Linears split. The fixed_mass K9 golden is re-recorded (fp32 reassociation; the per-row pass
    patched back in reproduces the old entry 99 / 99); blob's R1 and T2 compiled code are byte-identical. T2's X5 cost
    is not in this encoder (0.1–0.2 ms of 0.8–1.7 ms per forward): the residual and the OTHER pass own it. The cost
    ablation's 432 MiB headroom (`889add9d`) does not reproduce at `c7b4d03e` with the desktop stopped (1,106 MiB);
    why is UNVERIFIED (the desktop's 811 MiB is the likely part). Under Amendment 5 (§7.9) the +5 % / +3 % lines no
    longer gate the launch (s ≤ 50 %, s from the paired benchmark, not these reads); the headroom line is F-XC-2's
    refusal, which now clears at this box state without the override; F-XC-3 still stands.
  - **F-XC-3 FIXED, and the launch status (2026-10-05, `gen3_reference_state_released_v1`,
    `measurements/x5_launchable_2026-10-05/`, `measurements/x5_paired_speed_2026-10-05/`).** The parity gate's
    EAGER reference ran the slot replica's own forward, which replaces the extractor's and the op's per-forward
    stashes; they outlived the gate, sized by its last bucket's rows (fixed_mass ≈ 0.38 MB per row), so a slot load
    read them as a new allocation. `decision.policy_reference` now runs under `forward_state_released`, which empties
    every per-forward attribute the forward replaced; no compiled code moves (blob and fixed_mass R1 + T2 identical
    before and after). The REAL production launch (no override; X26 heads, regime A, `--eval-freq 400000`): every
    slot load grew ≤ 1,024 B (20 at startup, 2 pool refreshes), the T2 stage's allocated memory 1,264 → 749 MiB,
    `UpdateFit` headroom **1,790 MiB** (declared 1,024: PASS), the R1 gate and the update-10 canary PASS, no NaN.
    **The paired ABAB benchmark (3 blocks per arm, desktop stopped, §7.4's reader): s = +16.7 %** (block pairs
    +17.1 / +16.4 / +16.9 %; update-cycle 54.43 → 63.53 s, of which `train_ms` 40.96 → 46.17 s), so the
    **matched-wall-time checkpoint is 12M** (15M / 1.167 = 12.85M). s ≤ 50 %: **fixed_mass is LAUNCHABLE under
    Amendment 5**, with no refusal overridden. The +5 % / +3 % budget lines still fail; under Amendment 5 they no
    longer gate.

### 3.7 The opponent pointer, and the A and B heads re-based

**One flat candidate list** (`design_q_head.md` §3):
- the active's move seats + OTHER_move;
- switch → each revealed bench mon;
- switch → each hypothesis seat;
- OTHER_species.

**Scoring and labels.**
- One shared scorer over candidate tokens, plus the candidate's log π as a logit bias (π detached, §3.2), then one
  softmax.
- Labels: the chosen candidate.
  - A move outside the seats → OTHER_move.
  - A switch to an unseen species not in the list → OTHER_species. Its identity is known at reveal, so the label is
    exact and training-only.
  - Non-choices stay masked (existing semantics).
- The gradient mode stays `detached`, as production, for the A/B. X20's `label_only` is a separate lever.

**Downstream cells.**
- The cells that consume α / β (`intent_*`, `pair_outcome_*`, `switch_branch`, `conditional_threat`) are re-expressed
  exactly: α_SWITCH = Σ of the switch candidates, β_j = α(switch → j) / α_SWITCH over the seats.
- β_OTHER is a feature, NEVER renormalised away.
- **OTHER as a switch target or a move outcome must be priced, not zero** (the first version's §3.3 and §3.7
  contradicted each other here). Today `switch_branch.py:252-263` and `intent_conditional.py:193` take
  `Σ_j β_j · cell_j`. With zero rows for OTHER:
  - its share of `e_high` and `e_pko` is dropped;
  - **`e_mult = 0` is the IMMUNE value**, so OTHER would read as immune to every move (the same failure as the old
    typeless-Hidden-Power bug); `conditional_threat.py:183`'s `e_type` likewise;
  - α mass on OTHER_move falls outside the `[:k]` slice and reads like SWITCH mass ("no outcome this turn") in
    `pair_alpha`, `intent_threshold`, `intent_move_cell` and `intent_conditional`.

  **U3 part 3 built OTHER's averaged rows for the EDGE families** (D1's `[low, high, crit, pko = 0, type_mult]` on the
  tail, never immune); the op's `out_cells` / `opp_p_ghost` stashes these consumers read still have six columns, so
  the 7th column there is U4's (it can reuse `ExtractorForward._other_edge_cells`' OTHER-mode D1 pass).
  The fix depends on §9 M3. Under option (c) OTHER has the tail-averaged rows, so every cell prices OTHER's share on
  today's semantics and P(Ghost | OTHER) is the tail marginal (linear, exact). Under a no-physics OTHER each cell would
  need an invented neutral value (`e_mult` = 1, `e_pko` = 0), which is itself unprincipled.
- **`seat_live`** is applied by `pair_alpha` but not by `intent_threshold`, `intent_move_cell` or `intent_conditional`
  (pre-existing, F-X5-15). U4 makes all four consistent, because OTHER_move makes closed seats common.

**A and B.**
- A: unchanged.
- **B: columns = the flat list**, centred under the flat α, label at the chosen column.
  - B now gets OTHER rows instead of masked misses, and tells switch targets apart: both declared limits of the pre-X5
    parameterisation go.
  - `build_ridealong`'s `requires opp_intent` and `RideAlongBatch.alpha_seat_nums` move to the new pointer's stash in
    the SAME unit, otherwise the X26 heads break.

**As built (U4, `gen3_x5_flat_pointer_v1`; `agents/model/flat_intent.py`; fixed_mass only, blob byte-identical).**
- **The list.** Columns for K seats: `[0, K)` the active's move seats (THE one order's), `K` OTHER_move (token: the
  active's E5 seat — so `fixed_mass` now REQUIRES `entity_tail_seats`, a registry `requires` and a constructor
  refusal), `K+1 .. K+6` a switch to their slot j (a revealed mon, or the hypothesis a hidden slot holds; live iff
  addressable, not the active, revealed-or-holding-a-hypothesis), `K+7` OTHER_species (its refined trunk token).
  Scoring: one shared 2-layer scorer over (token ⊕ both team pools ⊕ a move / switch KIND one-hot), plus the candidate's
  log π — π_m (0 revealed), OTHER_move's log-mass, a hypothesis's log π (0 revealed), OTHER's log-mass — DETACHED at
  the source and again in the head (M10); a masked candidate is −inf; a row with no live candidate (padding only) is
  left all-zero so the CE cannot be NaN. Built from its own private seed (`FLAT_INTENT_INIT_SEED`) out of
  `IsolatedLinear`s.
- **α / β retired.** They are still CONSTRUCTED and still see SB3's orthogonal re-init (both draw from the global RNG),
  then `Gen3DualHeadMaskablePolicy._build` drops them (`retire_superseded_intent_heads`) BEFORE the optimizer is
  built: no state_dict key, no optimizer slot, every non-X5 initial byte equal to blob's (pinned:
  `hypothesis_set_test`). A standalone extractor (the delivery-graph / viewer tools) keeps them unused.
- **Labels** (`flat_intent_targets`, the ONE function the loss, B and the readers call; no Rust change — the intent
  label already carries the switch-in species num and the TRUE typed Hidden Power num): a move in the seats → its seat;
  a typed Hidden Power label → a REVEALED Hidden Power's seat (num 237, priced as its typed mixture; before U4 that
  click was a masked α row); a move that is an OTHER_move MEMBER (`beyond`) → OTHER_move; a revealed switch → its slot;
  a hidden switch-in → the hypothesis slot holding its species, else OTHER_species when it is in the tail. Masked:
  non-choices, a label on a dead candidate (the `reach` rule), and a choice outside every candidate's support
  (Struggle, a learnset gap, a species outside V) — counted as `flat_unmodeled_rate`, never guessed (F-X5-34).
- **The loss and the metric.** One CE over the list (`instrumented_ppo/flat_intent_fold.py`, static, traced
  `fullgraph=True`), `--intent-label-bot-weight` as in the blob fold; the set-valued partial credit is superseded (the
  OTHER_species label is that statement made exact). **`opp_intent/other_label_rate`** (F-X5-8) = of the opponent's
  CHOICES, the share labelled OTHER; split `other_move_label_rate` / `other_species_label_rate`; pooled and per
  opponent class. On the K9 golden buffer at cold start: 0.19 (moves 0.26, switches 0.16), unmodeled 0.
- **The cells, re-expressed exactly.** `compat_intent_logits`: α = [K seats, OTHER_move, log α_SWITCH] (a GUARDED
  logsumexp: the plain one is NaN in gradient on a row with no switch target, and `threshold_probs`,
  `IntentMoveCell` and `IntentConditionalMoveCell` do not detach α), β = [six slots, OTHER_species]; softmax of each
  is the flat distribution's own numbers. Every operand gains OTHER's column (`FlatConsumerOps`, `append_other` —
  LOUD when a stash lacks it): OTHER_move's seat-axis column is the op's per-candidate cells on the FULL move axis
  contracted with `FixedMassMoves.other_u` = π_m·[beyond] / Σ_beyond π (`pair_cells` / `pair_in` incl. the eight
  status coordinates, `pair_type_mult`, the c2 operands; num tables via `pair_outcome.seat_num_table` /
  `seat_in_set`) — E_tail[f(m)], the move-side twin of OTHER_species' `other_tail_probs`; OTHER_species' `out_cells`
  column is the OTHER-mode D1 pass (the tail-averaged defender, never IMMUNE, P(KO) nulled) and its `opp_p_ghost` is
  `other_tail_probs @ SPECIES_IS_GHOST`. Pinned against independent per-move oracles (MOVE_ACCURACY, MOVE_PHYS, the
  per-move type chart) in `flat_intent_test`.
- **`seat_live` consistent (F-X5-15).** `threshold_probs`, `IntentMoveCell` and `IntentConditionalMoveCell` take
  `seat_live` (the seats' meaningful-K gate + OTHER_move's liveness) and mask α by it, as `pair_alpha` does — under
  fixed_mass only, so blob is byte-identical (F-X5-35).
- **B re-based** (`FlatOppEffectEnsemble`): columns = the flat list; a move seat scored from its move ID, a switch
  target from its SPECIES (revealed or hypothesis), OTHER_move / OTHER_species from learned vectors; centred under
  the flat α; regressed at `flat_intent_targets`' column (OTHER rows included). `RideAlongSpec.opp_flat_k`
  selects it; still built from the private seed, outside `policy.optimizer`, every input detached — pinned
  bit-identical to learning by `ridealong_update_test`'s fixed_mass arm.

### 3.8 Versioning, Rust, gates

**Flag.**
- Add **`--belief-tokens {blob,fixed_mass}`**, STRUCTURAL, default `blob` until the A/B rules.
- Bump `MODEL_CONFIG_VERSION` from the code's current value by one, with a `_migrate_config` default to `blob`. Add a
  `ModelFlag` row on all five surfaces, `requires` = `t0_species_prior`, `move_belief_mode`, `opp_intent` and (added
  in U2) `opp_belief_slots`: the presence BCE that trains δ_θ and BeliefHead's re-targeted set BCE both ride
  `--opp-belief-aux-coef`, so without it δ_θ would never train. BUILT (U2): config v136; `'blob'` joins the
  registry's OFF spellings (`flag_registry.OFF_STRINGS`), so `requires` binds only `fixed_mass`; the production
  mirror records `belief_tokens: "blob"`, so a fresh `fixed_mass` arm is REFUSED by the arch-surface guard unless it
  passes `--allow-nonproduction-arch` (the A/B arm's consent).
- **No `ARCH_SIGNATURE` bump in U3 (ORCHESTRATOR, 2026-10-04).** Within `fixed_mass`, U3 changed the forward's
  meaning without one; a U2-era fixed_mass checkpoint fails a strict load only through the new `hypothesis_marker`
  key. **PENDING (legacy manifest D-L1, provisional):** the planned clean break at X5 adoption DECOUPLES the version
  floor from the signature — tying them would refuse the A/B and X26 runs themselves — so nothing here assumes a
  signature bump at adoption.
- **No `ARCH_SIGNATURE` bump while both arms must build at one commit.** The bump comes with the loser's deletion,
  through `snapshot._DEAD_FEK_*`, as the P11b deletions do.

**Rust.** No runtime observation or label change.
- The label-writer guard is **DONE** (F-X5-3, `680edc36`, `gen3_label_lookup_guard_v1`): every Rust label writer
  returns an `Err` (a FAULT) on a lookup it cannot make, and species match by dex num. It never fired (0 skips over
  34,001 measured episodes).
- The dex-row table's generator uses the existing encoder.
- The obs-golden linchpin and the Rust observation-parity gates are untouched by construction. Rerunning them is the
  check.

**Gates the build touches.**
- `flag_registry_test`, `flag_requires_test`, `derived_toggle_resume_test`.
- `tier_contract_test` (new modules declare tiers).
- `delivery_graph_test` + `MODULE_GRAPH_TOKENS`.
- `belief_label_only_gate_test` (new belief stashes join `_BELIEF_SUPERVISION_KEYS`).
- `learner_lifecycle_gate`.
- `extractor_compiles_test` (slow) and the compile-regions golden.
- `checkargs` (X5 is a typed lever in the A/B).
- New: the per-site class-E I1 / I2 tests and class-M / class-S tests (§3.5); the fixed-size construction's tests
  (fp64 reference, the residual bound, k = 0 and k = n structural branches, ∂BCE/∂τ = 0, the tie order); the dex-row
  byte gate and real-state cross-check.
- On adoption: the mirror sync, `mode_flag_doc_gate`, `arch_tables_test`, the K9 golden (§6), ARCHITECTURE.md and
  CHANGELOG in the same commit.

### 3.9 Departures from `design_q_head.md` §1 / §3, and why

| spec | this note | why |
|---|---|---|
| presence = `min(1, k·q)`, OTHER = k − Σ hypotheses | **logistic fixed-size marginals** σ(a + τ), Σ = k exactly (LML; I-projection) | `min` loses mass whenever a weight caps, and that mass lands in OTHER. The first version's iterative capped πps fixed the sum but set π = 1, which gives an infinite BCE on a wrong capped species (0.23 % of cold-start decisions, MEASURED). The logistic form keeps Σ = k, never reaches 0 or 1, and gives the BCE its exact logit. |
| "≈ 29 → ≈ 31 tokens", +2 | 61 → 62, +1 | The production sequence is 61 (event seats, measured). OTHER_move re-uses the active's E5 seat. |
| OTHER embedding learned + out-of-list average | the same; **OTHER's physics is §9 M3** (recommended: today's averaged construction on the tail) | A no-physics OTHER leaves 57–74 % of the hidden mass unpriced (§3.3). |
| (unspecified) the learned delta's input | Deep-Sets pool over REVEALED tokens + global, at T0, zero-init | The tier order forbids reading refined tokens at T0. The zero-init keeps cold start = the Smogon prior's marginal. |
| (unspecified) belief loss | set BCE; no Hungarian matching | Exchangeable slots; a proper score for the marginals; OTHER calibrated by linearity. |
| (unspecified) presence in max-type reductions | declared per class (§3.5); class M is §9 M2 | I1 and I2 cannot both hold for a max. |
| (unspecified) π's gradient | detached into every policy / critic use | §3.2 (review M10). |

### 3.10 Alternatives considered and rejected

1. **Keep the blob (status quo).** §1.3's evidence; and Q / search need concrete opponent columns.
2. **Per-slot hypotheses (6 slots × top-k each).** Breaks exchangeability, since gen 3 has no team preview. It
   duplicates species across slots against Species Clause, and multiplies tokens by k.
3. **Determinized worlds in the forward** (Foul Play / POMCP style: N sampled teams, average the network). Costs N×
   forwards, and is exposed to strategy fusion (Frank & Basin 1998; Long et al. 2010). Kept for SEARCH (conditional
   Poisson sampling from π), not the network.
4. **Slot Attention / DVRL latent beliefs.** They solve discovery and belief learning that we do not need (§2.1–2.2).
5. **Implicit belief through memory (Metamon).** No concrete candidates for the opponent pointer or Q.
6. **Renormalising hypotheses (no OTHER).** Hides blindness and miscalibrates by construction.
7. **Capped πps (Tillé).** Infinite BCE on a wrong capped species (§2.5, review M1).
8. **A larger hypothesis budget.** Now quantified (§3.3) and offered as §9 M3 option (a).
9. **Hand-defined ROLE tokens.** Rejected by the owner (2026-10-02): roles must be learned.
10. **A torch-side synthetic-row builder.** A second encoder (§3.4).

---

## 4. Role calibration (owner's addition; analysis only, no role tokens)

### 4.1 Roles, from Smogon only

- A role ρ is "carries move m".
- The role set is DATA-DRIVEN and pre-registered: every move whose Smogon expected carriers per team,
  Σ_s usage(s)·P(m | s) from `priors.moves`, is ≥ 0.25.
- The named examples (Spikes, Rapid Spin, Baton Pass, Wish, Heal Bell, Explosion) are reported if they pass that bar.
  No hand list.
- P(ρ | s) is the Smogon per-species marginal.
- **Multi-move roles are out of scope.** Smogon publishes no per-set data, and the pool may measure but never ship as a
  prior.

### 4.2 Metrics (both arms, plus the Smogon prior alone as a third column)

At each decision, the presence mass for role ρ is

M_ρ = Σ_revealed P̂_i(m) + Σ_hyp π_h·P̂_h(m) + π_OTHER·P̄_tail(m)

- P̂ is the network's own move posterior. Revealed moves are pinned at 1.
- **For the blob arm**, the hypothesis terms are replaced by its hidden-slot move presences, Σ_hidden P̂_slot(m). That
  is the blob's own belief about m among unseen mons.
- **For the prior column**, T0 → the fixed-size construction → Smogon P(m | s), so the learned deltas' contribution is
  separable.
- N_ρ is the number of opponent team members whose TRUE moveset holds m.

The four reads (each a per-run statistic; cross-arm inference is §7.4's per-seed rule):
- **R1, mean calibration** (Van Calster et al. 2016, level 1). Δ_ρ = mean(M_ρ − N_ρ), stratified by reveals r = 1…5.
  Within a run, a DESCRIPTIVE 95 % CI by cluster bootstrap over battles (Field & Welsh 2007). Summary: the
  usage-weighted mean |Δ_ρ| over roles.
- **R2, count Z.** Σ(N − M) / √Σ Var, with Var = Σ p(1 − p): CONSERVATIVE under the fixed group mass (§2.5).
  Reported, and never the sole verdict.
- **R3, substitute double-counting.**
  - A substitute pair (s₁, s₂) is pre-registered from Smogon only: both have P(m | s) ≥ 0.5 for a shared role m, and
    their teammate lift is < 1, i.e. they co-occur less than chance.
  - Rows: decisions where exactly one of the pair is in the true unseen set and neither is revealed.
  - Read: the rate at which BOTH have π ≥ 0.25 (π for X5; the fixed-size marginal of BeliefHead's posterior for the
    blob, so the two are comparable). Also the continuous excess π₁ + π₂ − 1.
  - **Rule 8:** a decision where either π lies in [0.245, 0.255] is EXCLUDED, and the excluded count is reported.
- **R4, OTHER calibration** (X5 only): OTHER mass against the realised count of true unseen species outside the list,
  by r. Reliability over OTHER-mass bins, plus the Spiegelhalter Z.

### 4.3 The reader, and where its rows come from

**The eval traces cannot carry this** (MEASURED, `rust_eval/traces.py:38`): Rust-core traces store `obs`, `logits` and
`values`, with no belief, intent or presence arrays. So the reader is a new offline CLI, `main.belief_roles`:
- **Substrate.** The fixed, committed Lane S policy-spectrum bank of about 20.7k RE-ENCODABLE turns
  (`../research_state/measurements/m5_laneS/`). It was built so that every architecture "including the discrete-token
  boundary X5" is read on the SAME turns (`program_rust_core.md`, Lane S ①).
  - The bank's core input logs carry both teams, so N_ρ is exact. VERIFIED (X5 Tier 0 F1; U7 reads both packed teams off
    every battle record, 0 label mismatches on the 20,712 decisions).
- **Both arms at ONE commit** through the flag, so the prober's arch-drift refusal does not apply. Forward passes only,
  on CPU.
- The bank's teams are pool teams. The owner's order is on-pool first; off-pool is a second column once a ladder bank
  exists (X8/X9).
- Its output never goes under `models/`.

Optional: a count DISTRIBUTION read with the nonrandomized PIT (Czado et al. 2009), if R1 flags a role.

**As built (U7, `main.belief_roles`; the choices §4.2 left open, each recorded in the Decision record).**
- **P̂ is MoveBelief's published TYPED posterior as sigmoid inclusion probabilities** (the supervised quantity), Hidden
  Power as ONE role (num 237; P̂ = Σ_t P(HP_t), the typed channels' presence), revealed moves pinned at exactly 1. Under
  `fixed_mass` the hypothesis terms read the hypothesis seat's own (species-specific) posterior times its π, and OTHER's
  term is Σ_tail π_s · P(m | s) with the network's own Smogon move prior (`build_move_prior_logits`, the E10 mixture's
  source; F-X5-32: OTHER's moves are parameter-free).
- **The prior column** prices a revealed mon by Smogon P(m | s) with its revealed moves at 1 and the moveset-exhaustion
  rule (4 revealed ⇒ every other move 0); its hidden part is the T0 fixed-size marginal × P(m | s).
- **R4 has a DERIVED OTHER for `blob` and the prior**: the same construction (Σ_tail π past the k-th in the one stable
  order) on their own presence, so the three columns are comparable; for `fixed_mass` the read's OTHER is checked equal to
  the model's `other_mass` (≤ 1e-4) or the read refuses.
- **R2's variance** sums p(1 − p) over every contributing indicator (a revealed mon, a hypothesis π·P̂, each tail species
  π_s·P(m | s)).
- **The role set is derived at READ time** from the facade (`roles.SOURCES`; a fresh-interpreter audit-hook test fails
  any other data file) and stamped (`role_set_sha256`): 28 roles and 2,798 substitute pairs on the corrected move
  prior AND the rating-weighted usage marginal (F-X5-41 and F-X5-47, both fixed 2026-10-04; sha `823b27be…`) — Spikes
  0.501 expected carriers, Rapid Spin 0.419 and Baton Pass 0.373 pass; Ice Punch (0.246), Wish (0.166) and Heal Bell
  (0.045) do not. With the move prior fixed but the usage still `Raw count` it was 29 roles / 3,030 pairs (`daba9995…`,
  Ice Punch 0.257); under the deflated move prior, 15 roles and 18 pairs (`4e3ab394…`).

---

## 5. The KL early stop: analysis kept for X28, NOT built

**Owner, 2026-10-03: OFF for the X5 A/B and for X26, not even as a guard.** "We need a more robust setup … 10 [epochs]
is probably too much and we probably need a dynamic controller arm for managing the number of epochs". That is X28,
the held-out-yield epoch controller, the named lever after X26. Nothing in this section is built in the X5 units; it is
kept because X28 needs the measured fire rates.

### 5.1 What the vendored stop does (`instrumented_ppo/ppo.py:541-556`)

- It checks ONE MICRO-BATCH's approx-KL, the k3 estimator (r − 1) − log r of KL[old ‖ new] (Schulman 2020) over 2,048
  rows, at the current parameters before that micro-batch's backward pass, against 1.5 × `target_kl`.
- At the production shape, 98,304 rows ÷ 2,048 = 48 micro-batches per epoch. With accumulation 32, each epoch is one
  full group plus one ragged group: **2 optimizer steps per epoch, 20 per update at E10**. `main.dose` counts the same
  way.
- A trip zeroes the open accumulation group and skips every remaining epoch. The critic, belief and intent losses share
  that loop, so they would lose those passes too.
- **Today it never fires:** `self.target_kl` is never set. `InstrumentedMaskablePPO` gets SB3's `None`
  (`model_build.py:528-545`). The recipe's `kl_controller.target_kl` (0.01) only feeds the KL→LR controller's band,
  [0.005, 0.02].

### 5.2 How often it would have fired: the sizing arms

Source: per-epoch `train/approx_kl_epoch_k` in the four arms' TensorBoard files, read 2026-10-03. Scratch scripts in
`/home/goodlad/.claude/jobs/9c36ca35/tmp/x5/c2/`. No per-micro-batch KL is logged.

| arm | fire rate, LOWER BOUND (an epoch-mean > 0.015 forces a micro-batch > 0.015) | first trip by epoch (median) | optimizer steps kept, of 20 (≤) | modelled fire rate, per-micro rule, c = 4–7 | modelled steps kept |
|---|---|---|---|---|---|
| A2 (N = 48, E10) | **0.488** (42/86) | 5 | 15.3 | 0.62–0.66 | 12.7–11.4 |
| A′ (N = 48, E10) | **0.407** (33/81) | 5 | 16.0 | — | — |
| B (N = 256, E10) | **0.506** (41/81) | 6 | 15.4 | 0.61–0.69 | 13.0–11.3 |
| C (N = 256, E5) | **0.383** (31/81) | 2 | 7.7 of 10 | 0.46–0.52 | 7.0–6.4 of 10 |

- The lower bound is MEASURED and exact on the open-loop, no-feedback replay. No epoch mean lay within 1e-6 of 0.015.
- The model is ESTIMATED: gamma-distributed micro-batch draws around the reconstructed KL path, with coefficient of
  variation c/√2048. c, the per-row coefficient of variation, has a plausible range of 4–7 from the age-1 probe.
- **KL is CONCAVE in steps, not quadratic:** epoch 0's first step already reaches 34–49 % of the final epoch's KL.
- At B with c = 4–7 the stop would keep about 5.7–6.5 effective epochs (57–65 % of the nominal dose); O8 found E5 vs
  E10 costs −10.56 pp U [−13.12, −7.94] at about 60 % of B's dose. That is why the owner kept it out of the A/B.

### 5.3 Couplings any future stop must fix (for X28)

- The KL→LR controller reads `train/approx_kl` = the LAST epoch's mean; after an epoch-0 trip it reads ≤ 0.001 and
  pushes the LR UP, a positive feedback loop.
- `main.dose` must count REALIZED steps, not nominal epochs (`dose.py:149-175`).
- `update_fit.dry_update` must run with `target_kl = None`.

---

## 6. The K9 learner golden re-bake

**What it pins today** (`training/learner_golden.py`, `learner_golden_test.py`, `designs/training/learner_gates.md`).
- Inputs:
  - a production-surface learner rebuilt from seeds, built at ONE torch thread by `build_learner` itself (F-X5-4,
    `0c25a1f4`);
  - a committed 64-row real Rust buffer (`learner_golden_buffer.npz`, sha `c6008684…`);
  - the golden recipe (2 epochs, micro-batch 16 × accumulation 3, LR 2.8e-5);
  - one eager fp32 CPU `train()` on 1 thread.
- Compared: initial and post-update parameter sha256, 41 per-group post-update hashes and 19 scalars, all EXACT.
- Keyed by torch version only, not by arch signature.
- Re-baked with `python -m agents.training.learner_golden record --reason …`, which appends a history row.

**Re-bake plan, and what proves it CORRECT rather than merely new:**
1. **X5 OFF is byte-identical.** With `--belief-tokens blob` (the default), the EXISTING golden passes unchanged at the
   X5 commit. This proves the flag constructs nothing and draws no global RNG when off.
2. **X5 ON** gets a SECOND golden entry (key `fixed_mass`), built through `build_learner(args = production + X5)`.
   Before it is recorded, these must hold:
   - losses, KL and gradient norm finite (K9(c));
   - KL and clip fraction in range;
   - **two processes with different `PYTHONHASHSEED`, both at 1 thread, produce identical hashes**;
   - the NON-X5 parameter groups' INITIAL bytes equal the blob arm's, if X5 initialises from an isolated generator.
     That needs the harness gap closed: store `init_group_sha256`.
3. **Independent references (fp64 numpy, on the golden's 64 rows):**
   - Σ team mass = 6 and Σ move mass = 4 to the §3.2 tolerance;
   - the fixed-size construction against a direct fp64 root-find;
   - ∂BCE/∂τ = 0 on every row (§3.2);
   - OTHER's bias is −1e9 iff its count is 0;
   - attention with log-π bias equals the π-weighted renormalised softmax;
   - I1 / I2 on class E, I1 on class M (§3.5).
4. **Not vacuous:**
   - every X5 parameter group's post-update hash differs from its initial hash;
   - each new loss key is logged and nonzero;
   - a TEETH test: perturb k or the OTHER mask, the golden must FAIL naming X5's groups; revert, and it passes.
   - Coverage: the buffer holds 29/64 rows with all six opponent species known, but only **2/64** with an opponent mon
     with 4 revealed moves, so the OTHER_move-masked case is thin. Assert per-case counts ≥ 2, or rebuild the buffer
     (`rebuild-buffer --reason`, which invalidates every entry).
5. **Harness hazard (F-X5-4) — FIXED 2026-10-03 (`0c25a1f4`).** A learner built outside `_one_thread` at 8 threads
   moved the INITIAL hash (SB3's orthogonal re-init is a LAPACK QR whose rounding follows the thread count).
   `build_learner` now enters `_one_thread` itself and restores the caller's count. The PRODUCTION trainer's fresh
   build had the same dependence: FIXED 2026-10-03 as F-X5-5 (`50fdfdc2`: `construct_fresh_learner` and the fresh
   fixtures build inside the ONE shared `utils.torch_state_guard.single_thread_build`), so a fresh X5 run's start no
   longer follows the core count. §7.4 still records the build thread count as a precondition check.

**As built (U6, 2026-10-04).** The second entry is `arms.fixed_mass` in `learner_golden.json`; the blob fields are
byte-for-byte unchanged (the X5 code is byte-identical there, and the blob test passes untouched). `learner_golden.ARMS` declares it:
- **Learner.** `production_args()` + `belief_tokens = fixed_mass`, built by `build_learner` at one thread. The
  perturbation is NAME-KEYED (`parity_probe._keyed_noise`: each parameter's noise comes from its own generator
  seeded by `sha256(seed:name)`). The order-keyed draw would move every parameter after the first added or retired one
  (`belief_slots` is parameter 149 of 254), so no shared group could be compared with blob's.
- **Buffer.** Its OWN file, `learner_golden_buffer_fixed_mass.npz` (sha `681dc6b0…` in the JSON). This is not because keys are missing:
  the observation spaces are identical. It is because the blob buffer's stored behaviour log-probs are the blob
  learner's: K9(b) of the fixed_mass learner on it reads max |Δ| 0.51. It uses the same rollout recipe at run seed 18, because seed 17 gave
  this arm no OTHER_move-dead row against item 4's ≥ 2 rule. The games differ from blob's, since the policy differs.
- **Recorded.** `init_params_sha256` + `init_group_sha256` (40 groups), `post_params_sha256` + `group_sha256`,
  17 losses, the fp64 reference, the K9(b) read and the coverage counts.
- **Proven** (`learner_golden_fixed_mass_test.py`, 16 tests; `learner_golden_threads_test.py`):
  - item 2: two processes at `PYTHONHASHSEED` 0 / 4242 and 1 / 8 BLAS threads give the same bytes, and every
    shared group's initial bytes equal blob's;
  - item 3: the fp64 reference, below; Σπ = k (species) and 4 − r (moves); π equals a direct numpy fp64 bisection
    to 1e-5; OTHER's log-mass is −1e9 iff its tail is empty;
  - item 4: every X5 group moved and the new loss keys are nonzero; the teeth below; coverage OTHER_species
    live 36 / dead 28, OTHER_move live 60 / dead 4, 0 rule-8 near-tie rows.
- **The fp64 reference** (`learner_golden_fp64.py`). One `micro_step` (region R1: every term of fold steps
  1–3a) at the seeded init over the judged rows, at fp32 and at fp64 (a float64 copy under `Fp64Mode`).
  - Rule 8: rows with a K9(b) tie margin under 2e-4 are excluded, 2 of 64.
  - Measured: terms ≤ 3.3e-7 abs and ≤ 1.3e-6 relative; seven key gradients ≤ 1.5e-6 relative L2.
  - Declared bars: 5e-6 + 1e-5·|t| per term, 5e-5 per gradient.
  - An fp32-only 1e-3 logit shift fails it. It is the SAME code at two precisions, so it is not independent.
  - The attention-equals-π-weighted-softmax and I1 / I2 checks of item 3 stay in `hypothesis_tokens_test` (U3);
    they are not repeated at golden level.
- **Teeth.** Each of four plants FAILS the fixed_mass golden, moving 14–15 losses and 35 groups, both X5 groups among them;
  none moves the INIT:
  - τ + 1e-2 in the construction;
  - the log-π key bias's sign;
  - OTHER's cells read at the next column;
  - OTHER_move masked out of the flat pointer.
  All four together leave the blob golden byte-identical. α's head + 1e-2 (blob-only) fails blob and leaves fixed_mass
  byte-identical.
- **Isolation.** With every set BCE's logits detached, δ_θ's bytes do not move in the real update; unplanted, they
  move. The arm + B (`--ridealong-opp 2`) equals the entry in every recorded group (initial and post) and every
  loss, so B is bit-identical to learning.
- **K9(b)** passes on the entry: max |Δ| 4.8e-7, excluded share 6.25 % (1 of 16 rows) against the 0.15 ceiling. With
  the probe ON the update is bit-identical to OFF. The share needed F-X5-44's fix: before it, 47 % of rows were excluded.

Not covered by the golden: compiled and CUDA numerics. Those belong to K8's regions, startup parity and the canary.
X5 makes no observation change, so the obs golden must stay green untouched.

---

## 7. The A/B design (registered; orchestrator decisions, 2026-10-03)

### 7.1 Decomposing the replicate floor (review M4 added the A − A2 row)

The script is at `/home/goodlad/.claude/jobs/9c36ca35/tmp/x5/vardecomp.py`; the A − A2 row and the pools are
`x5_revision_2026-10-03/scripts/m7.py`. Both read the banked
`../research_state/measurements/m5_sizing/results/meters_read.json`.

**The model.** For two runs' untaught-meter (U) levels, E[Δ²] = 2σ²_run + σ²_meter,Δ. σ²_meter,Δ comes from the
meter's own team-clustered, paired CI, which contains game and team noise only. The meter is deterministic at seed 0,
concurrency 1, and has reproduced exactly five or more times (ledger L20461 and later). So everything the CI omits is
RUN-level, which is UNDERSTANDING rule 19's distinction.

| pair | depth | games | Δ (pp) | meter SE(Δ) | meter share of Δ² | σ_run (pp) |
|---|---|---|---|---|---|---|
| **A′ − A2** (seed 1002 vs 1001, same pin, M5 core, opponent N0@24M) | 8.06M | 600 / team | −4.90 [−6.21, −3.60] | 0.66 | **1.8 %** | **3.43**, 95 % CI [1.53, ∞) (1 df) |
| **A − A2** (SAME seed 1001, pins `f9349f95` vs `277f318f`) | 8.06M / 8.13M | 600 / team | **−11.35 [−13.73, −9.00]** | 1.21 | 1.1 % | **7.98** (1 df) |
| W_b − W (old lineage, opponent `ai_v9_29@24M`, L19064) | 75M | 200 / team | 3.69 | 1.40 | 14 % | 2.41 (1 df) |
| B − A2 (same seed, N 256 vs 48; sensitivity only, assumes no N effect) | 8M | 600 / team | −2.71 [−4.50, −0.94] | 0.91 | 11 % | 1.80 |
| pooled: A′−A2 + W_b−W / + A−A2 | — | — | — | — | — | **2.97** (2 df, [1.55, 18.7]) / **5.21** (3 df, [2.95, 19.4]) |
| SD of the four E10 levels A, A2, A′, B (34.5, 45.8, 40.9, 43.1) | 8M | — | — | — | — | **4.84** (3 df; includes B's N lever and A's pin) |

**Is A − A2 run noise? Probably mostly, and nothing can separate it.** What differs between the two runs:
1. **The pin.** 12 commits. Learning-relevant: `f6b32f5a` (eval-dump isolation) — A's KL→LR controller lost 4 of 82
   readings (≈ 5 %), A2's none. Both runs' LR paths have the same range (3.0e-4 → 4.32e-4). The others are not
   learning-relevant on this recipe: `7d550baa` (R1's levers, which production does not use), `a52ed9b4` (eager
   fall-back detection, rounding only), `64aef25d` (checkpoint cadence, unchanged at N = 48 for a fresh run),
   `01612190` (a hook refactor, identity-tested), and docs.
2. **A crash-restart in A2** at 4.52M from `checkpoint_4000032` (a K9(b) FATAL, F-SZ-8): 0.52M steps re-collected on a
   new trajectory.
3. **CUDA nondeterminism from the first update.** The two runs' first approx-KL reads differ in the 8th digit
   (0.0026600221 vs 0.0026600107) with identical first clip fractions (D-9 declared this). A seed id does not
   reproduce a GPU run.
4. 81 vs 87 updates (A2 overshot by 0.85 % and repeated the post-restart steps).

So A − A2 is a replicate-like pair with a ≈ 5 % controller difference and one restart on top. Its −11.35 pp is either
an unlucky draw from a σ ≈ 3–5 distribution or evidence that σ_run is larger than the seed pair says. **Consequence
for the design:** every sizing table below carries σ ∈ {2.5, 3.43, 4.8} rows (4.8 ≈ the four-arm SD, between the
pooled 2.97 and 5.21), and §7.4 also reports 5.21 and 8.0.

**Facts in the replicate pair.**
- All 8 teams moved the SAME way (per-team Δ −2.2 to −8.0, SD 2.0). That fits a run-level strength shift, not team
  noise.
- **Guard meter (G-A) on the same pair: Δ −0.83 [−4.79, +3.13].** Its meter variance (SE 2.0) exceeds the observed Δ²,
  so G-A's run component is undetectable at 1,200 games. That is weak evidence that part of U's run variance is
  specific to its one opponent.

**Verdict.**
- **Training-run variance dominates: about 98 % of the replicate gap at 8M and 600 games per team; meter noise is
  about 2 %.**
- More games, a GSPRT on games, or CUPED all reduce the 2 %, and buy nothing. The remedies are only a meter whose RUN
  floor is smaller, averaging within-run snapshot jitter, and more seeds.

**Caveats.**
- Each σ estimate rests on 1–3 df. UNDERSTANDING rule 3 calls a floor "the MAX pairwise |Δ|"; by that rule the floor
  is now 11.35 pp, not 4.90.
- The U floors are against DIFFERENT opponents (F-X5-20).
- No untaught read exists at 15M or 25M on any lineage, so σ_run at the A/B's depth is UNMEASURED.
- **No run floor exists on the HEAD-TO-HEAD scale at all.** §7.4's simulation borrows the U-scale values (G-1).

**FINDING F-X5-6.** The sizing study and the learner battery ruled non-inferiority with "95 % CI lower bound > −3.69",
where the CI was the meter's game-and-team CI. That CI omits the run term. Their nominal α was not their real α. This
design puts σ_run in the SE.

### 7.2 Sizing on the untaught meter (now a reported secondary)

With known σ, true Δ = 0, one-sided α and power 1 − β, the margin is δ = (z_{1−α} + z_{1−β})·√(2(σ²_run + σ²_m)/K),
where K is seeds per arm and σ²_m = 0.22 pp² per run at 600 games per team. ESTIMATED from §7.1.

**Untaught meter (U), α = 0.05 one-sided, power 0.8, δ (pp):**

| σ_run | K = 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 2.5 | 8.9 | 6.3 | 5.2 | 4.5 | 4.0 | 3.7 | 3.2 | 2.8 |
| **3.43 (8M seed pair)** | 12.2 | 8.6 | 7.0 | 6.1 | 5.4 | 5.0 | 4.3 | 3.8 |
| **4.8 (four-arm SD)** | 17.0 | 12.0 | 9.8 | 8.5 | 7.6 | 6.9 | 6.0 | 5.4 |

- With σ estimated inside the experiment (t on 2(K − 1) df) the margins are wider still at small K.
- **GPU-hours** (sequential, one GPU, N = 256): blob at 15M ≈ 2.5 h, X5 ≈ 2.6 h (ESTIMATED from §3.6, op term
  pending U8); 3 seeds per arm ≈ 15.3 GPU-h, 5 ≈ 25.5, 8 ≈ 40.8.
- **15M vs 25M.** 25M costs 1.67× per run. It beats 15M at equal GPU-h only if σ_run(25M) < 0.77·σ_run(15M). The only
  depth evidence (3.43 at 8M, 2.41 at 75M, 1 df each) does not show that. **15M.** Risk named: if X5's benefit appears
  only late, 15M under-reads it; in a non-inferiority design that falls on adoption, not safety.

### 7.3 P0, a CPU pre-study on banked checkpoints: a PLANNING input only

The orchestrator pre-registered the head-to-head as primary (§7.4), so **P0 no longer selects the meter** (this removes
the review's M6, a one-df selection step). P0 estimates the H2H run floor σ_h for the power table, and the snapshot
jitter, before any A/B GPU.

**Inputs.** `sizing_A2_n48_e10_s1001`, `sizing_Ap_n48_e10_s1002`, `sizing_B_n256_e10_s1001`, and
`sizing_A_n48_e10_s1001` (the cross-pin twin), read-only: their finals and their 8.0M checkpoints.

**Reads, all CPU:**
1. **H2H round-robin of the four finals,** mirrored pairs on the Rust eval core, 1,000 pairs per edge (meter SE
   ≈ 1.1 pp). A2 vs A′ is the pure seed edge; A vs A2 the cross-pin edge; the B edges are sensitivity edges.
   **σ̂_h² = (Δ̂² − SE²_meter) / 2** per edge: the meter's own variance is SUBTRACTED, never left in.
2. **U on the 8.0M checkpoints of A2 and A′** (600 per team). U(final) − U(8.0M) is snapshot JITTER at about one update
   of separation.

**Tool.** No CLI plays checkpoint against checkpoint offline today: `main.anchors` takes external opponents only. U0
builds the H2H-cross CLI from `sprt.py` + the Rust eval core; the same CLI is the A/B's primary meter.

**Use.** σ̂_h updates §7.4's power line before registration is frozen. It changes NO boundary and NO margin: those are
fixed now, and the boundaries do not depend on σ (estimated variance).

### 7.4 Registered design and decision rule

**Arms and order.**
- Two arms, `blob` and `fixed_mass`, at ONE commit through the flag. Production recipe at N = 256, with NO KL stop
  (owner, 2026-10-03) and the X26 ride-along heads ON in both (owner, X26 by continuation).
- Seed ids 1001, 1002, … in order, the same ids in both arms for bookkeeping only. **Seeds do not pair runs:** two GPU
  runs of one seed diverge from the first update (§7.1 item 3), and X5 draws its extra parameters from the init stream.
  The analysis treats the arms as independent; the cross design needs no pairing.
- Checkpoints every 1M from **10M** (the speed rule's matched-wall-clock checkpoint and the jitter read need them).
- Arm order alternates by seed (X1 B1 B2 X2 X3 B3 …), to block box drift.
- **Preconditions:** the build thread count is recorded in `metadata.json` (`init_num_threads`) and equal across every
  run (F-X5-4 / F-X5-5; the production fresh build is pinned to one thread since `50fdfdc2`, so this is a check); runs launched from the commit that adds `main/train/init_num_threads_test.py` record the key (before it nothing wrote it, so an earlier run's absent key reads UNKNOWN, not 1); U8's cost budget passed. A run that breaks a precondition is
  INCONCLUSIVE.
- Through the training agent, `gpu_lock` throughout.

**Primary meter: the mirrored head-to-head CROSS.** At each look every X5 seed's final (15M) snapshot plays every blob
seed's, 1,000 mirrored pairs per cell (≈ 2,000 games; meter SE ≈ 1.1 pp per cell). Cells: 9 at look 1, 25 at look 2,
64 at look 3 (only new cells are played). h_ij = X5_i's win rate against blob_j, in pp. CPU cost per cell UNMEASURED
(U0 measures it).

**Statistic.** Δ̂ = mean_ij(h_ij) − 50. Row means R_i (each X5 seed against the blob population) and column means C_j
(each blob seed against the X5 population). V̂ = (s²_R + s²_C)/n on **df = 2(n − 1)**, n seeds per arm.
**t = (Δ̂ + δ) / √V̂.** Under the additive model h_ij = Δ + a_i − b_j + noise this is the two-sample t on the arms'
run strengths, and it uses every cell. The paired-by-seed alternative (d_i = h_ii, n − 1 df) was simulated and is
dominated (power 0.53 against the cross's 0.58 at σ = 3.43, both without the futility stop).

**Margin δ = 3.5 pp** (≈ 24 Elo at 50 %), one-sided **α = 0.05** (orchestrator: X5 is the intended end state; the
test exists to catch HARM, and a false "non-inferior" costs at most δ, which X26 then measures anyway).

**Boundaries.** O'Brien–Fleming on z for three looks at information fractions 3/8, 5/8, 1 (z = 2.785, 2.157, 1.705
for α = 0.05), mapped to t at the same nominal p (Pocock 1977; Jennison & Turnbull 1991). The plain mapping gave a
worst simulated type-I of 0.0525, so the constant is **scaled ×1.02** (worst over the grid 0.048):

| look | seeds / arm | df | NON-INFERIOR iff t ≥ | nominal one-sided p |
|---|---|---|---|---|
| 1 | 3 | 4 | **5.761** | 0.0023 |
| 2 | 5 | 8 | **2.683** | 0.0139 |
| 3 | 8 | 14 | **1.874** | 0.0410 |

Rule 8: a t within 1e-9 of a boundary is NOT a crossing.

| outcome at a look | rule |
|---|---|
| **INCONCLUSIVE** (any look) | the look's inputs are incomplete or invalid: a run that did not reach 15M or exhausted its restarts, a broken precondition, an H2H cell with < 1,000 completed pairs or timeouts > 25 % of attempted battles. The look is RE-READ after the repair (a failed run is replaced by the next seed id, never dropped), never interpreted. |
| **NON-INFERIOR** | t ≥ the look's boundary. |
| **FUTILITY STOP** (looks 1–2, non-binding) | Δ̂ ≤ −δ. Verdict NOT DETECTED, labelled INFERIOR if the upper one-sided 95 % bound of Δ (t on the look's df) is < −δ. Non-binding: the type-I figures below hold whether or not it is followed (FDA 2019 §V.A). |
| **CONTINUE** (looks 1–2) | neither: add seeds to the next look (+2, then +3 per arm). |
| **NOT DETECTED** (look 3) | t below 1.874. Blob stays; X5 goes back to the owner with the read. Never "equivalent" (rule 6); INFERIOR label as above. |

**Simulated operating characteristics** (MEASURED by simulation; `x5_revision_2026-10-03/out/gs_sim3.log`,
`gs_extra.log`; 200,000 replications per cell; σ = per-run SD on the H2H scale; cell noise 1.1 pp; type-I at
Δ = −δ; with the futility stop):

| σ | type-I | power at Δ = 0 | NON-INFERIOR at look 1 / 2 / 3 (Δ = 0) | E[GPU-h], Δ = 0 | E[GPU-h], Δ = −δ |
|---|---|---|---|---|---|
| 2.5 | 0.046 | **0.81** | 0.03 / 0.34 / 0.44 | 33.6 | 26.1 |
| **3.43** | 0.045 | **0.57** | 0.02 / 0.19 / 0.36 | 34.4 | 26.1 |
| 4.8 | 0.047 | **0.36** | 0.01 / 0.11 / 0.25 | 33.6 | 26.1 |
| 5.21 (pooled with A − A2) | 0.045 | 0.33 | — | 33.2 | 26.1 |
| 8.0 (A − A2 alone) | 0.047 | 0.19 | — | 31.3 | 26.1 |

- **Type-I holds at every σ:** ≤ 0.048 without the futility stop, ≤ 0.047 with it, also with a seed × seed interaction
  of 1.5 pp; at small σ (0.5–1.5) it is conservative (0.02–0.046), because the cell noise then dominates V̂.
- **Power at σ = 3.43 is 0.57, BELOW 0.6, at the full 41 GPU-h ceiling.** A fixed single look at n = 8 would give only
  0.61; the sequential design costs ~3 pp of power and saves ~6 GPU-h when X5 is truly equal and ~15 when it is
  truly δ worse. The futility stop costs ~1 pp of power.
- **The margin-against-budget trade** (same design, same cost): δ = 4.5 pp (≈ 31 Elo) gives power 0.94 / 0.76 / 0.51
  at σ = 2.5 / 3.43 / 4.8; δ = 5.5 pp (≈ 38 Elo) gives 0.99 / 0.89 / 0.66. **The owner chooses (§9).**

**Secondary meters (REPORTED, never gated).**
- The untaught meter on each final, with §7.2's known-σ margins as context.
- **The outside panel** (frozen pool snapshots, bots, SmallRL), REPORTED with a pre-declared **HARM flag**: if X5's
  panel point estimate is worse than blob's by more than **2δ = 7 pp**, the owner is told before adoption, whatever the
  primary says. **Why not a gate:** a second non-inferiority test must pass too, so the joint power is roughly the
  product. At n = 8 per arm and δ = 3.5, a panel of 300 games per run has NI power 0.54 / 0.44 / 0.32 at
  σ = 2.5 / 3.43 / 4.8, and 0.75 / 0.56 / 0.38 at 1,200 games (MEASURED by formula, `m7.log`): gated, the joint power
  at σ = 3.43 would fall from 0.57 to about 0.25–0.32.

**Purpose metrics (the adoption gate; review M8).** Read on the Lane S bank, on-pool primary:
1. Opponent-intent log loss on the common event space (the realised action's probability; a blob miss is reported as
   its own column, never floored into the score).
2. Species presence Brier (resolution), with class-wise calibration (Gupta & Ramdas 2022).
3. OTHER calibration (R4).
4. Role calibration R1 / R3.

- **Inference is ACROSS SEEDS.** Each run gets ONE value per metric (its mean over the fixed bank); the arms are
  compared by a two-sample t over seeds on 2(n − 1) df. A within-run, battle-clustered bootstrap (Field & Welsh 2007)
  is DESCRIPTIVE only: it is conditional on the run and on the fixed bank, and omits the run term exactly as F-X5-6's
  CI did.
- **Fixed sequence with a group-sequential boundary.** Metric (1) is tested only after strength is NON-INFERIOR, at the
  look where that happened, against **the same look's t-boundary** (5.761 / 2.683 / 1.874), not at full α: testing a
  secondary at full α at the crossing look inflates its type-I (Hung, Wang & O'Neill 2007; Tamhane, Mehta & Liu 2010).
- **X5 is ADOPTED iff strength is NON-INFERIOR and (1) improves past that boundary;** (2)–(4) are reported.
- **As built (U7, `python -m main.belief_roles read | infer`).** Every read is on the Lane S bank in its fixed order,
  re-encoded at the reader's commit (gate ① byte-checked), CPU forwards through the strict loader.
  - **(1)'s common event space:** the opponent's realised action at the decision (the trackers' `IntentLabel` at the
    viewer's next entry — the label training aligns to the row) as a move by num, every Hidden Power ONE event (its type
    is a set property, not a choice), or a switch-in by species. Each arm's heads induce P(event): a move = Σ of α over
    the seats naming it; a switch to s = α_SWITCH · Σ_j β_j · c_j(s) over β's legal slots, c_j = slot j's CONTENT (a
    revealed slot's species; a hidden slot's BeliefHead species posterior over V under `blob` — the content β's training
    target is addressed by). Under `fixed_mass` the FLAT pointer is read (U4; α / β are retired there): one softmax
    over [K seats · OTHER_move · six slots · OTHER_species], a slot's content its revealed species or its hypothesis,
    OTHER_move's the renormalised tail π_m / Σ π_m over its members, OTHER_species' `other_tail_probs`. An event no
    candidate names is a MISS (`intent_miss_rate`, with a breakdown: inside / outside the active's learnset, switch),
    never floored; the log loss is over the covered rows. A Struggle label (forced, no PP) is excluded and counted.
    MEASURED on cold / perturbed fresh checkpoints: `fixed_mass` misses 0 % (OTHER covers every modelled event), `blob`
    4.9–9.0 % (moves outside its six seats; never a switch) — so the two arms' log losses are over different row sets,
    and the miss rate must be read beside it. **Superseded as the adoption gate by Amendment 3(b)'s CONDITIONAL form
    (§7.7(b) "As built", `intent_logloss_conditional`); this form stays reported, descriptive only.**
  - **(2)** the team Brier Σ_V (π − y)² per decision (and the set log score, the Murphy split on fixed bins, class-wise
    mean calibration); `blob`'s π is BeliefHead's hidden-slot-mean logits through the same fixed-size construction.
  - **Per run:** the `per_run` block (on-pool) holds one value per metric; `infer` takes the X5 runs' and the blob runs'
    JSONs and the look's boundary, and refuses mixed banks / role sets / arms, a repeated checkpoint, n < 2 or zero
    variance. Rule 8: a row within 1e-6 of a selection boundary the read depends on (`near_tie_rows`; the derived
    OTHER's k-th gap; blob's E4 seat cut) is excluded from that read and counted.

**Speed rule (X5 non-inferior but slower).**
- **s** = (median update-cycle wall of X5's updates) / (median of blob's) − 1, pooled over each arm's seeds; an
  update-cycle is the wall between consecutive `train/` update records; updates whose window contains an eval cycle or
  a restart are excluded (C-1's reader rule).
- **s ≤ 5 %:** adopt on the rule above.
- **5 % < s ≤ 15 %:** ALSO require non-inferiority at MATCHED WALL-CLOCK: the same cross with each X5 run at its
  checkpoint at or below 15M/(1 + s) (the nearest 1M checkpoint at or below, so it is deterministic), on the same
  boundaries. This is the strength-per-GPU-hour rule of `design_q_head.md` §8.
- **s > 15 %:** stop and report before any meter read, as an optimisation unit.

**Deletion.** On adoption, the blob path is deleted in the next deletion unit, and the signature is bumped (§3.8). The
winning arm's seed-1001 run CONTINUES as the X26 baseline (owner).

### 7.5 Amendment 1 (2026-10-04, registered BEFORE any arm launched): the blob arm runs first

These deviations from §7.4 were approved by the owner on 2026-10-04 and written down before the first launch. Anything
in §7.4 not named here still stands: the meter, the statistic, δ = 3.5 pp, the boundaries, the outcomes, the secondary
meters, the purpose metrics and the X26 continuation.

**Why.** The fixed_mass arm can't launch today. It fails all three U8 budget lines and hits four startup refusals
(`d7120efa`; F-XC-2..5). The blob arm is production and can launch now, so the owner chose to keep the GPU busy on
it and grind the fixed_mass fixes afterwards (owner, 2026-10-04).

| §7.4 says | Amendment 1 | what protects the comparison |
|---|---|---|
| Arms alternate by seed (X1 B1 B2 X2 …), both at ONE commit | **The blob arm's look-1 seeds 1001, 1002 and 1003 run FIRST, in that order, all pinned to one commit P_blob.** P_blob is the first main commit carrying F-X5-47's fix and this amendment. **The fixed_mass seeds run later, pinned to a later commit P_x5** after the cost fixes. Look 2's added seeds alternate again as §7.4 says. | **Blob-path identity across the two pins** is a PRECONDITION; breaking it makes look 1 INCONCLUSIVE, and the blob seeds re-run at P_x5. At P_x5: (i) the K9 golden's `arms.blob` entry is byte-identical to P_blob's; (ii) the sha256 of every `data/pokemon/` file the runtime reads is identical, and so are the obs golden, `OP_SEMANTICS` and the reward and critic constants. A blob-moving GIGO fix between the pins needs the owner's word and voids look 1's blob side. **`data/` is frozen while any arm is live:** a pin isolates code, not data. Box drift without interleaving moves WALL time, not strength at matched steps. Strength is read at the 15M checkpoint, so drift reaches only the speed rule (next row). |
| Speed rule: s from the arms' own update-cycle walls | **s comes from a PAIRED BENCHMARK at P_x5:** blob and fixed_mass alternate in one session on a quiet box (desktop stopped, T23). Both use the same recipe and the X26 heads, and the read follows §7.4's reader rule. It runs as ABAB… blocks, at least 3 per arm, near the weekly quota reset. | The arms' own walls come from two commits and two box states, so their ratio would confound X5's cost with drift. F-XC-6 already shows such drift: the Rust core step rose 6.6 → 10.0 ms with no X5 code, cause unknown. The 5 % / 15 % thresholds and the matched-wall-clock checkpoint follow from this s, unchanged. |
| Preconditions include "U8's cost budget passed" | **U8 gates the fixed_mass seeds' launch only.** | The blob arm IS production, so no cost question applies to it. The fixed_mass seeds don't launch until U8 passes or the owner revises the budget. |
| (not stated) | **The snapshot-ladder updater is OFF in every arm on both sides:** `--snapshot-ladder-games 0` (F-ED-18). | Its `--promote` children outlive the trainer's SIGTERM at ~90 % CPU each (F-G-10). Those orphans would steal Rust-core CPU unevenly across arms. No arm reports or quotes a `ladder.json` ELO; the primary is the h2h cross, played offline at each look. |
| (not stated) | **The h2h protocol is FROZEN for the A/B.** Every cell of every look is played on one h2h engine build and one game protocol, recorded on the look's eval-ledger rows. A change to either re-plays the look's cells under one protocol. | The eval review's A/B rule (M3); the ledger's protocol version on every row. |
| (not stated) | **No CPU eval lane beside a live arm.** | It would contend with the Rust core. None is built today (eval U4 is paused), so this is a check, not a change. |

**The experiment's surface.** Each blob seed types these levers on top of `--arch production` (the production ARCH
surface + `recipe.fresh`, N = 256, no KL stop). This is a list of levers, NOT a launch command: the training agent
builds and validates the argv per `TRAINING_RUN_SOP.md` §1 (`main.checkargs`, `--dry-run`).
- `--belief-tokens blob` (explicit; it is also production's value);
- `--steps 15000000`, `--seed <1001 | 1002 | 1003>`, run name `rb_x5ab_blob_s<seed>`;
- the X26 ride-along heads, exactly X26's registered argv: `--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5
  --ridealong-opp 5 --ridealong-rnd-variants all`;
- `--snapshot-ladder-games 0`;
- `--checkpoint-every-steps 1000000` (§7.4's 1M checkpoints from 10M; this flag also saves the earlier ones).

The fixed_mass seeds later type the same list with `--belief-tokens fixed_mass` and run names `rb_x5ab_fm_s<seed>`.
`metadata.json`'s `init_num_threads` must be equal across all six look-1 runs (§7.4 precondition).

### 7.6 Amendment 2 (2026-10-04, registered BEFORE any fixed_mass or oracle seed): two ORACLE REFERENCE arms, the ceiling

**Why** (owner, 2026-10-04: *"an arm of our current experiment with the discrete hypothesis tokens because it tells us the ceiling"*). Non-inferiority tells us whether the hypothesis tokens cost strength. It does not tell us how much any belief COULD buy. A network that is simply TOLD the opponent's team is the ceiling every belief representation is trying to approach, so the A/B carries it as a reference. The arms are backlog X32 (`EXPERIMENT_BACKLOG.md`), moved into this experiment.

**The arms.** Each is the production `blob` architecture plus `--oracle-reveal <level>` (flag name provisional). The Rust encoder writes the revealed facts into the OBSERVATION's opponent team block, encoded exactly as an in-battle reveal is. So they enter the SHARED TRUNK and reach every head through it. They are not a side input to the policy or the value function (owner, 2026-10-04). Obs dims are unchanged; `off` is byte-identical (a parity test).
- **Oracle-species** (`species`, team-preview semantics): all six opponent species known from turn 1, nothing else.
- **Oracle-full** (`full`): species + four moves + item + ability + nature + EVs + IVs, as already-known. Item and ability are the orchestrator's addition so it is the whole set; the owner to confirm.

Run names `rb_x5ab_oracle_sp_s<seed>` / `rb_x5ab_oracle_full_s<seed>`; the same levers and recipe as the blob arm (§7.5's surface + the flag), 15M, 6 h restart interval, ladder updater off. Pinned to P_x5, under the same blob-path identity precondition as the fixed_mass seeds; the reveal must be inert when off. Reveal is SYMMETRIC in training: the oracle's self-play opponents see its team too, the later-gen framing.

**How they are read: REPORTED, NEVER GATED.** §7.4's decision rule for adopting the hypothesis tokens is unchanged: its arms, statistic, δ, boundaries, α, outcomes, purpose-metric sequence and speed rule. No α is spent on the oracle arms.
- **Cells.** Each oracle seed plays every blob seed and every fixed_mass seed in the same mirrored cross (1,000 pairs per cell), at look 1 (3 seeds per oracle arm). Later looks add oracle seeds only if look 1's oracle interval is too wide to read.
- **The ceiling.** C = (oracle − blob) in pp, the same cross statistic as Δ, with its interval over seeds, reported per oracle arm. Oracle-full − oracle-species isolates the SET's share.
- **Headroom captured.** H = Δ / C: the fraction of the oracle's advantage the hypothesis tokens recover. It is reported ONLY when C's lower 95 % bound is above the replicate floor; a ratio over a denominator near zero is noise. Otherwise the read is "the ceiling is within the noise: belief representation has little strength leverage at this budget", itself a finding.
- **Caveat stated in advance.** The theoretical value of information bounds what any belief can buy (Blackwell 1953). A TRAINED oracle can fall short of it (shortcut learning, optimisation), so C estimates the ceiling at this recipe, not a strict bound. The old-era critic-only privileged channel (2026-09-10) bought no value resolution, so a small C is a live possibility.
- **Pre-registered predictions** are written in the launch's ledger entry before the first oracle seed.

**Cost and order.** The reveal build (both levels in one unit, ≈ 1–1.5 agent-days: Rust encoder, goldens, obs benchmark, a fuzz run) joins the fixed_mass fix queue. GPU ≈ 2.3 h per seed, 3 + 3 seeds ≈ 14 GPU-h on top of the A/B's ≈ 41 GPU-h ceiling (the owner's own addition). Oracle-species runs before oracle-full.

**As built — the `species` level (2026-10-04).** The flag is `--oracle-reveal {off,species,full}` (default `off`); `full` is described after this paragraph's list. `encoder::oracle::Level` is an enum, and the Rust `Spec.oracle_reveal`, `protocol.ORACLE_REVEAL_LEVELS` and the registry row's choices are one list each, so a level is a variant and a slot builder, not a new surface.

*The mechanism is the ENCODER's, per chain.* `src/rust_sim/src/encoder/oracle.rs` builds an `Oracle` per episode and side from the OTHER side's packed team (`team::unpack`, species ids by `to_id`; `Oracle::new` is called in `rust_env`'s `Env::start`, and at pool startup for every team of the table, so a team the reveal cannot render fails there). The `Oracle` rides on the side's `SideStream` (an `Arc`, shared by a fork) and is read by exactly one place, `slot::team` for the opponent block. The side's reading, view, trackers and legality are the chain's own, untouched. The opponent block is then:
- the mons the viewer has SEEN, at their reveal-order positions with the bytes `off` writes;
- after them, one slot per unseen opponent mon, in dex-num order (stable; the order the belief labels already give an unseen mon), each the row `encoder::hypothesis::hypothesis_slot(species)` writes — the encoder's own row for a never-seen mon, produced by the same `slot::populated_slot` a real first appearance goes through, so the two cannot render one mon differently;
- a mon that play then reveals LEAVES the tail and joins the seen prefix, matched by DEX NUM (a forme shares its base species' num; one oracle entry is consumed per revealed mon, so a duplicate is neither double-counted nor dropped). A revealed mon the oracle team does not hold is a FAULT.

*Rejected alternatives.* Pre-populating the reading's opponent roster, or injecting synthetic `|poke|` lines, would put the unseen mons in the reading, the view, the recency / seen tracker, the HP belief and every label that reads `reading.opp`, each a place where "seen" could leak. Rewriting the row after the encoder would duplicate the slot layout. The encoder-only mechanism touches exactly the cells declared below.

*Every field.*

| cells of an unseen opponent slot | value | why |
|---|---|---|
| species num + 6 base stats; type ids; ability block | the species' dex row; the Smogon top-2 ability prior (or the one ability of a one-ability species) | derived from the SPECIES alone |
| `species_known`, HP fraction, recency ×3, protect odds | 1.0 each | the true state of a mon that has never been on the field: populated, full HP, "never seen" (saturated), full Protect odds. It is NOT marked seen, acted, hit or active |
| item block, status one-hot, sleep / toxic counters, 4 move slots, spread block (`spread_known` 0), Hidden-Power block (`hp_revealed` 0, probs 0), sleep belief, last action, trapped, maybe-trapped, active flag | exactly 0.0 | hidden until play reveals them |

`oracle::SPECIES_SLOT_CELLS` declares all 122 cells (it tiles the slot) and `oracle::check_species_slot` THROWS when a built slot breaks a rule (bit comparison; `-0.0` and NaN fail): the producer's guard, run on every slot at episode start. The other blocks are left alone: both teams' seen slots, our team, the two active contexts, the global block, the board block, the event window, and the pair-history block (a tail slot's pair cells are the never-interacted values, which are bit-for-bit an absent slot's: `SAT_LUT[0]` is 0 and the recency cell is 1.0).

*Labels.* The belief labels read the row's `species_known` and the reading's seen list. With the row stating all six, no slot is believed: `belief_species` and `belief_moves` are all PAD, and the belief aux loss on an all-PAD minibatch is its declared no-op (`None`, no NaN: pinned on the REAL core's labels by `oracle_reveal_integration_test.py`). `known_moves`, `hp_type`, `item` and the three spread label families cover every STATED slot — seen first, then the tail — in the encoder's own order, because `labels::*` call the same `Oracle::tail` the encoder calls. The consumer's guard: the belief writer refuses a row whose stated-species count differs from what the reading plus the oracle give (a producer / consumer drift would shift every label one slot). The intent label is unchanged: a switch to a not-yet-seen mon keeps `SWITCH_SLOT_NONE`, because the switch-slot target stays in the seen frame (a refinement, listed below).

*Training and eval.* The reveal is SYMMETRIC by construction (each side's chain is given the other side's team, so the trainee and a policy opponent served by T2 each see the other's species); scripted bots read the view, not the row, so they are unaffected. `RustEnvDecl.oracle_reveal` is the run's recorded mode, and `build_eval_core` reads it from the same declaration, so the in-loop eval plays at the run's mode. The fork arm is refused with a reveal (`oracle_reveal_vs_fork_arm`: its successor rows come from search chains the reveal is not given to).

*Record.* `ModelVersion.oracle_reveal` (config v137, a `ModelFlag` row of class `resume_immutable`: no module, no weight, so `check_compatible` and the ARCH surface do not read it), recorded in `model_config.json` and `metadata.json`, inherited by a flagless resume, a flip refused by `check_oracle_reveal`, printed by `main.checkargs` and the launch banner. `main.h2h`, `main.anchors`, `main.play`, `main.belief_roles` and `main.policy_spectrum` REFUSE a checkpoint recording a non-`off` mode (`agents.model.oracle_reveal`), because they build observations without the reveal.

*Gates (counts in the ledger entry).* INERT WHEN OFF: `rust_env/tests/oracle_reveal_test.rs::off_is_inert` pins the obs, mask and every label column over 27,333 real decisions to a digest recorded on `e0d56693`, the commit before the build; `species_bytes_are_pinned` pins the `species` level's own bytes so a later level cannot move them. DIFFERENTIAL: `species_differs_from_off_only_in_the_declared_cells` plays `off` and `species` side by side (one staging, one policy) and, at every decision of both sides, requires every cell outside the unseen tail bit-identical, the seen slots identical, the tail equal to `hypothesis_slot` of the true unseen species in dex-num order, the opponent block's species multiset equal to the true team (turn 1 included), and the label columns equal on the seen slots and covering the tail. Unit tests: dex-num order and shrink-on-reveal, a changed forme still leaves the tail, Species-Clause duplicates, a revealed mon off the oracle team, the producer guard's teeth per block, a real Forecast battle through a forme change.

*DEFERRED (both levels).* (1) ~~the per-SIDE reveal in the h2h engine (§7.7(a)'s one-sided cells)~~ BUILT 2026-10-05, §7.7(a) "As built"; (2) the intent head's switch-slot target for a not-yet-seen mon (stays NONE); (3) search chains (`search_driver`, the Rust env's search, the fork arm) are not given the reveal; (4) the Python encoder has no reveal, so the prober and the Lane S bank cannot re-encode an oracle run's states; (5) the GPU-side checks, in the first minutes of the training launch: the K9 learner-golden gate and the compile region on the learner (the obs dim is unchanged, so no new graph is expected), the first update's belief and hp-type / item / spread losses finite on CUDA, the T2 policy-opponent slot reading oracle rows, and `behaviour/` near-tie counts (K9(b)) against the blob arm's.

**As built — the `full` level (2026-10-04).** `--oracle-reveal full` tells the observation the opponent's whole SET from turn 1, as already-known facts: the four moves, the item, the ability and the spread (nature, EVs, IVs; `spread_known` 1) of EVERY opponent mon, seen or not. It is the same mechanism with the level a variant of `encoder::oracle::Level`; `off` and `species` are byte-identical to what they were (`off_is_inert` and `species_bytes_are_pinned`).
- *An unseen mon's slot* is `full_slot(set)`: the row the encoder writes for an OWN mon of that set at full HP — the same `slot::populated_slot`, fed a `PMon` built like an own mon's reading (`PMon::from_species`, then the item, the ability, the moves at full PP with a bare Hidden Power typed from the set's declared type or its IVs exactly as the owner's request spells it, and the spread through `PMon::backfill_spread`, the SAME backfill an own mon takes from its packed team: the nature lower-cased, `serious` if none) and a view whose `spread_known` is true. The slot writer's spread block and `hp_revealed` now follow the view's `spread_known` (true for an own mon, false for an opponent under `off` and `species`, true under `full`), so no byte of `off` moved. The Hidden-Power block reads `hp_revealed` 1 and probs 0, as for an own mon (the moves determine the type).
- *A SEEN mon's slot* is the reading's, with only the facts play has NOT revealed written on top (`Oracle::overlay`, called by `slot::team` for the opponent block): the item only while the reading's item is the unknown sentinel (a revealed item, and a consumed or removed one — the reading holds `None` — stay as play left them); the ability only while none is revealed (a revealed or changed one stays); the true moves the reading has not seen are learned at full PP, an observed move keeps its slot and its tracked PP, a bare `hiddenpower` the reading learned stands for the set's typed one (no fifth move), and a TRANSFORMED mon (it shows its target's moves) gains none; the spread is told (it is never revealed in play). So no stale preview value fights a live one.
- *Declared cells.* `FULL_SLOT_CELLS` (18 blocks; tiles the slot): the set's facts (item, ability, moves, the spread block) are free, `hp_revealed` is exactly 1, and every other cell is the pristine never-seen mon's, as at `species`. `check_slot(Level::Full, …)` throws on a break (teeth per block; a `species` slot fails the full guard on `hp_revealed`, a `full` slot fails the species guard on `item`).
- *Verification.* The differential `full_differs_from_off_only_in_the_opponent_block_and_tells_the_true_set` plays `off` against `full` over real battles and, at every decision of both sides, requires every cell outside the opponent block identical to `off`, a seen mon's non-fact cells identical to `off`'s, an unseen mon's non-fact cells equal to the `species` row, and — the independent oracle — every fact block equal to the OPPOSING CHAIN'S OWN-TEAM slot of the same mon in the same battle at the same decision (the encoder's own row for a mon whose set it knows): for an unseen mon everywhere, for a seen mon wherever play has not revealed the fact (item known in play → the reading's, ability likewise, moves = the revealed slots untouched + the true remainder, spread and Hidden-Power block = the set's). It found two real defects before it passed: the nature was not lower-cased like an own mon's (every nature modifier read 1.0), and a typed Hidden Power was resolved to the bare id (the owner's request types it from the IVs); a third, a consumed item overwritten by the true one (`None` is not "unknown"), was found by the same differential. Unit tests (`encoder::oracle`): the tail slot as an own row, the overlay case by case (sentinel, revealed, consumed item; unrevealed and revealed ability; an observed move's PP; the bare Hidden Power; a transformed mon), the guard's teeth and the cross-level refusals, and the overlay's refusals. `full_bytes_are_pinned` pins the level's own bytes.
- *Labels* are the `species` level's (every stated slot labelled with the true moves, item, spread; nothing believed): with the moves visible in the row, the known-slot move labels are now trivially predictable, a reported consequence, not a defect. `opp_switch_slot` is unchanged.
- *Deferred at this level:* the same list as `species` (below), plus: a transformed seen mon is not overlaid, and a seen mon's PP for a move play revealed is the VIEWER's tracked value (Pressure or Spite can make it differ from the truth, exactly as under `off`).

### 7.7 Amendment 3 (2026-10-04, registered BEFORE any fixed_mass or oracle seed and before ANY purpose-metric read)

**(a) The oracle arms are played in TWO evaluation modes (owner, 2026-10-04).** Every oracle cell of §7.6 is played twice, mirrored pairs both times:
- **One-sided clairvoyance (PRIMARY for C):** the oracle sees its opponent's team; the blob or fixed_mass seed plays exactly as trained, with no reveal. This is the pure value of the information.
- **Both-sided reveal (SECONDARY):** the non-oracle seed is ALSO handed the oracle's team, through the same observation reveal at the same level. **Caveat, stated in advance:** that seed never saw a revealed team in training (a full reveal at turn 1 is out of its distribution), so its play here may be degraded by the unfamiliar input, not by the information. The mode is DESCRIPTIVE only and is never read as "information parity".

The h2h engine sets the reveal per SIDE, which is a build requirement. C, oracle-full − oracle-species and H = Δ / C are computed from the one-sided cells; the both-sided cells are reported beside them.

**As built — the per-side reveal (2026-10-05).** `python -m main.h2h play|play-many --oracle-reveal-mode {off,one_sided,both_sided}` (`src/main/h2h/reveal.py`).
- *The levels are never typed.* A plan declares ONE mode, and each cell's `(p1, p2)` levels follow from it and the two checkpoints' RECORDED `oracle_reveal`: `one_sided` = the oracle side at its own level, the other side `off`; `both_sided` = both sides at the oracle's level (the other side is told the ORACLE's team). Typed refusals before anything loads: an oracle checkpoint under `off`, a reveal mode over a cell with no oracle side, an oracle against an oracle (`RevealModeError`); a side about to play at a level other than what its checkpoint and the mode require (`RevealLevelMismatch`, checked per cell against the core it plays on).
- *The mechanism is the training core's.* The Rust spec key `oracle_reveal` takes a per-side pair `[p1, p2]` (`core::spec::Reveal`; a symmetric value is still written as the string, so every training spec's text is unchanged), and `Env::start` gives each side's chain the OTHER side's team at its OWN side's level — the same `Oracle::new`, the same declared field sets. `build_eval_core(oracle_reveal=(p1, p2))` passes it (the player is p1); the engine declares one eval core per (player group, opponent group, levels). An oracle checkpoint's weights load exactly like any other (the reveal is an observation mode), and the other offline tools (`main.anchors`, `main.play`, `main.belief_roles`, `main.policy_spectrum`) keep refusing one.
- *Stamped.* Each mode is its own eval protocol (`gen3_eval_protocol_v1_h2h_oracle_one_sided` / `..._both_sided`; `design_evaluation.md` §0b.2), so its rows are their own regime and never pool with `off`; every revealed row carries `compute.oracle_reveal = {mode, player, opponent}`. A family pins one protocol (§0c rule 6), so the A/B's oracle cells are a request and a family PER MODE, separate from the blob × fixed_mass look's.
- *Verified.* (i) `off` is the engine of before, bit for bit: a 24-game play recorded at `c7b4d03e` (every action, every margin and log-prob bit of both sides) and its ledger row (every non-volatile field, regime id `adfdbee824c9eb0a` on that box's pool) reproduce exactly at the build; the play is pinned in `play_reveal_integration_test.py`. (ii) Rust: a split core's side rows (obs, mask, every label column) are bit-identical to the SYMMETRIC (training) core of that side's level, 13,606 rows over species and full in both seats (`a_per_side_reveal_writes_each_side_exactly_as_the_symmetric_core_of_its_level`; it fails when both sides take p1's level). (iii) Real h2h games: one-sided moves the oracle side's turn-1 decision in every game and leaves the other side's bit-identical (on the first wave, whose batches are alike — a later game's turn-1 batch can differ, so it is excluded, never tolerated), in either seat; both-sided moves both, and the oracle side's turn-1 decision equals its one-sided one; every row is stamped. Unit tests: the level matrix and every refusal (`reveal_test.py`).
- *Not built:* an oracle-vs-oracle cell (refused; §7.6's oracle-full − oracle-species is read from each against blob); GPU per-cell timing at a reveal (the CPU engine is unchanged; DEFERRED to the GPU owner).

**(b) The purpose metric's OTHER-mass bias is removed (found by the owner, 2026-10-04).** §7.4's metric (1), as built (U7), averages −log P(realised event) over each arm's OWN covered rows. That favours `blob` twice:
- **Selection on the outcome:** blob is scored only on rows where the answer was nameable; its misses (4.9–9.0 % cold) are dropped.
- **Structural over-certainty:** even on shared rows, blob's probability sums to 1 over its named candidates while fixed_mass reserves mass for OTHER. Equal skill reads worse for fixed_mass by ≈ −log(1 − P(outside)).

The two arms are also TRAINED on different questions: blob's intent head has its unnameable rows masked; fixed_mass is supervised on OTHER. **Replacement, decomposed so neither arm is scored on what the other cannot express.** Per row, E_row = blob's named candidate set (its move seats + switch targets on the common event space).
1. **Metric (1), the adoption gate, now:** the CONDITIONAL log loss −log [P(e) / Σ_{e′ ∈ E_row} P(e′)], on rows whose realised event e ∈ E_row, with EACH arm's probabilities renormalised over E_row. Same rows, same support, both arms. Tested exactly as before: only after strength is NON-INFERIOR, at that look's t-boundary, inference over seeds.
2. **Coverage, REPORTED:** for fixed_mass, the mass outside E_row against the observed outside frequency (blob's miss rate on the same rows). Calibration-in-the-large plus a reliability curve on fixed bins: does fixed_mass predict "something blob cannot name" as often as it happens? For blob, its miss rate.
3. **The as-built unrenormalised log loss and the miss breakdown stay REPORTED, descriptive only.**

Rule 8 applies to any row whose renormalisation denominator is within 1e-12 of 0 (excluded and counted). The U7 reader change (`main.belief_roles`) joins the Monday-night queue: a new `intent_logloss_conditional` + `intent_coverage` block, with tests that a planted over-certain arm cannot win metric (1).

**As built — the conditional metric (2026-10-06, `main.belief_roles`, `src/main/belief_roles/eset.py`).**
- *E_row is the blob checkpoint's support on the common event space, decided by membership:* the HP-collapsed move events of the E4 seats α can name (finite logit, seat num > 0), plus switch-in s iff α_SWITCH is finite and some β-legal slot's content supports s (a revealed slot its species; a hidden slot BeliefHead's posterior, which supports all of V). On a blob row "e ∈ E_row" and the as-built `covered` are the same predicate; the reader raises if they ever differ.
- *A fixed_mass run is scored on EVERY blob run of the look, and its value is the MEAN* (Decision record, 2026-10-06, the orchestrator's decision superseding the seed-id pairing first built). Each blob run's set is read on the same bank rows; the read reports the conditional log loss and coverage per reference and their mean, and the mean is the run's `per_run` value (a None on any reference makes it None). A blob read writes its set beside its JSON (`<label>.erow.npz`, content-hashed, bank-stamped); a fixed_mass read takes them with `read --reference <fm label>=<blob>[,<blob>…]` (labels of the same invocation or `.erow.npz` paths; a repeat is refused). Without any, the fixed_mass read's conditional value is None. Every read records `eset_reference` (`own` / `all_blob_mean` with each reference's checkpoint sha256 / `none`). `infer` on `intent_logloss_conditional` refuses unless every control read is on its OWN set and EVERY treat read references EXACTLY the control group's blob checkpoints (a missing, extra or repeated reference, or sets that differ across fixed_mass runs, is refused). So every fixed_mass run is scored over the same collection of supports the blob arm is, as in the cross, and the t stays the unpaired two-sample t of §7.4.
- *Each arm's FULL distribution on the dense event space* (moves by num, switch-ins at 1000 + species) is built from the same heads and contents the as-built read scores. Blob: Σ α over the seats; α_SWITCH · Σ_j β_j c_j(s). Fixed_mass: the flat pointer's seats, OTHER_move's renormalised members, the slots, OTHER_species' tail. The reader asserts the dense value at the realised event equals the scored log-probability on every covered row (≤ 1e-9 nats).
- *The read:* labelled, non-Struggle rows. Rule 8 excludes a row whose reference seat cut is a near-tie (the set is ill-determined), a row whose arm read is (`read_tie_arm`, as before), and a row whose denominator Σ_{E_row} P ≤ 1e-12, each counted. An in-set event to which the arm gives exactly zero mass is counted (`n_zero_event_mass`) and turns the value to None: never floored, never dropped. Split by move / switch / opponent class.
- *Coverage (REPORTED, `intent_conditional.coverage`):* the arm's mass outside E_row against the realised outside indicator on the same rows. It gives calibration-in-the-large (mean mass − frequency, battle-clustered CI), a Spiegelhalter-type z, and a reliability curve on fixed bins (0, .01, .02, .05, .1, .2, .3, .5, 1]. For blob the frequency is its miss rate.
- *`per_run` adds* `intent_logloss_conditional` (THE adoption-gate value; `infer` tags it), `intent_set_miss_rate`, `intent_outside_mass` and `intent_outside_citl`. `intent_logloss` / `intent_miss_rate` stay, descriptive. Read schema `gen3_belief_purpose_read_v2`, so `infer` refuses a mix with v1 reads.
- *Blob's own mass is not always 1 on its own set:* its α keeps SWITCH finite on rows where no β slot is legal, so that mass names no event. The as-built read charged it to blob; the renormalisation removes it, as the definition requires. That is why blob's conditional value sits slightly below its as-built one.
- *Tests that fail on revert* (`conditional_test.py`, `infer_test.py`, `belief_roles_integration_test.py`):
  - THE planted test: an over-certain arm (all mass on E_row, the same 3:1 ratio inside it) and an honestly calibrated arm of equal discrimination read EQUAL on metric (1), while the as-built form hands the over-certain arm a strict win.
  - The renormalisation by hand; rule 8 at the denominator and at the reference tie; a zero-mass in-set event giving None.
  - Coverage calibration by hand; the dense distributions against the scored log-probabilities.
  - The mean over every blob set: per-reference values by hand, the run's value their mean (not the pooled rows, not one reference), a None on any reference making it None.
  - `infer`'s refusals: a missing, extra or repeated reference, sets that differ across fixed_mass runs, the old seed-id form, a blob read not on its own set.
  - End to end on a bank slice: a blob read's set feeding a fixed_mass read.
- *Smoke on real checkpoints (NOT the registered read; one seed per arm, no inference; 2026-10-06, on-pool, CPU, ~2 min).* `rb_x5ab_blob_s1001` vs `rb_x5ab_fm_s1001` finals (15.05M steps each), the fixed_mass run scored on blob s1001's set ONLY (taken before the mean-over-the-look decision; re-read through the mean-over-references code with that one set, the value is identical):

  | | blob s1001 | fixed_mass s1001 |
  |---|---|---|
  | metric (1), conditional (rows scored) | **1.900** [1.863, 1.940] (11,475) | **1.752** [1.714, 1.798] (11,370) |
  | as-built log loss (rows), descriptive | 1.902 (11,475 of 12,049) | 1.980 (11,944 of 11,944) |
  | set miss rate / outside frequency | 4.76 % | 4.81 % |
  | mean mass outside E_row; CITL | 0.17 % (α_SWITCH with no target); −4.60 pp | 10.27 %; **+5.46 pp** [+4.93, +6.00] |

  Rule-8 exclusions: 105 fixed_mass rows (its own near-ties), 0 at the denominator; 225 Struggle rows each. The as-built form and the conditional form order the two runs in opposite directions. The fixed_mass run reserves about twice the outside mass that is realised, so its coverage is UNDER-confident about blob's set.

**(c) Why the tokens are kept even at equal strength (owner, 2026-10-04).** Interpretability is now a design criterion: when the cost is similar, take the architecture that makes the model's state more observable. Concrete hypotheses, a flat opponent pointer and named moves are directly readable, which should let the probing infrastructure be simplified. This does not change the adoption rule; it is why "non-inferior" is the right question.

### 7.8 Amendment 4 (2026-10-04, before look 1 is read and before either seed starts): blob seeds 1004–1005 run early

**What changes.** Look 2's two additional blob seeds, 1004 and 1005, run NOW, at P_blob = `e5e660dd` (the same pin as seeds 1001–1003), queued after the oracle seeds. Amendment 1 said look 2's added seeds alternate between arms again; for the blob side they do not.

**Why** (owner, 2026-10-04). The GPU would otherwise idle from about Monday 08:00 until the fixed_mass fixes land. Look 1 says CONTINUE with high probability (NON-INFERIOR at look 1 is 0.01–0.03 by §7.4's simulation), so these seeds are almost certainly needed.

**Why it is legitimate.** The seed list is fixed in advance and does not depend on any result, so running a seed early is not optional stopping. Their pin is the blob arm's own, so they are the most comparable runs possible. They are NOT read at look 1: look 1 stays at 3 seeds per arm (1001–1003). If look 1 stops for futility, they are unused, at a cost of about 4.6 GPU-h and about zero quota.

**Surface.** Identical to §7.5 (`rb_x5ab_blob_s1004` / `_s1005`, 6 h restart interval, ladder updater off).

### 7.9 Amendment 5 (2026-10-05, before any fixed_mass seed and before any read): run at up to 50 % slower; strength read at BOTH matched steps and matched wall-time; adoption is the owner's decision

**Why** (owner, 2026-10-05): *"As long as it isn't more than 50% slow, let's run the experiment and I want strength at matched steps and matched wall-time. I know this is a late change but I am now being asked to understand what to do given the performance cost, not just the strength."* The fixed_mass cost work may not reach the +5 % budget (§3.6), and a cost-free strength verdict no longer answers the owner's question.

| §7.4 / §7.5 said | Amendment 5 |
|---|---|
| The fixed_mass launch waits until U8's cost budget passes (§7.5) | **The fixed_mass seeds launch once s ≤ 50 %**, with s the paired-benchmark slowdown (Amendment 1). The startup refusals (F-XC-2 headroom, F-XC-3 slot load) must still be FIXED, never overridden: a budget waiver is not a refusal waiver. |
| Speed rule: s ≤ 5 % adopt on steps; 5–15 % also require matched wall-clock; > 15 % stop and optimise | **Both reads are made and reported at EVERY s ≤ 50 %.** (1) **Matched steps:** §7.4's cross at 15M vs 15M, unchanged. (2) **Matched wall-time:** the same cross with each fixed_mass run at its checkpoint at or below 15M / (1 + s), the nearest 1M checkpoint at or below, so it is deterministic, against the blob runs at 15M. Each read gets its own verdict at the SAME look-wise t-boundaries (5.761 / 2.683 / 1.874) and outcome table. |
| X5 ADOPTED iff strength NON-INFERIOR and purpose metric (1) improves (§7.4) | **No automatic adoption.** The owner decides, with both strength verdicts, purpose metric (1) (Amendment 3's conditional form, still tested in sequence after a NON-INFERIOR matched-steps verdict, at that look's boundary), s, and the GPU-hours. |
| (not stated) | **Error accounting:** neither read is a rescue for the other. Reporting both with no "either suffices" rule spends no extra α. A decision that REQUIRED both would be an intersection-union test, also with no α penalty. A decision that accepted EITHER would inflate the type-I error and is not allowed. |
| (not stated) | **Reading the wall-time read honestly:** at s = 50 % it compares fixed_mass at about 10M steps against blob at 15M. A wall-time deficit with steps non-inferior means "costs strength per GPU-hour NOW", a cost question that an optimisation or the encoder refactor could change. It is not evidence about the representation. |

Everything else in §7.4–§7.8 stands. The oracle reference arms are unaffected (reported, never gated).

**Status (2026-10-05, MEASURED):** s = **+16.7 %** (paired ABAB, block pairs 16.4–17.1 %;
`measurements/x5_paired_speed_2026-10-05/`), so the matched wall-time read takes each fixed_mass run's **12M**
checkpoint against blob at 15M. The F-XC-2 and F-XC-3 refusals are FIXED and a real production fixed_mass launch
passes every startup check and its update-10 canary with no override (`measurements/x5_launchable_2026-10-05/`).
The fixed_mass seeds may launch once their pin P_x5 carries this fix (§7.5's blob-identity precondition is checked
there).

---

## 8. Risks, open gaps, build plan

### 8.1 Risks

| risk | mitigation |
|---|---|
| The design is underpowered at σ = 3.43 (0.57) and badly so if A − A2 reflects σ_run (0.19–0.33) | Stated (§7.4); margin against budget is the owner's call (§9). P0's σ̂_h tells us which regime we are in before any GPU. |
| A reduction that does not weight by presence ("counts as a whole mon") | The 41-site census; class E: I1 / I2 per site; class M: I1 + regression + declared semantics (§3.5) |
| OTHER unpriced (57–74 % of the hidden mass) | §9 M3; the recommendation prices it on today's semantics |
| The hypothesis attacker op adds unbudgeted cost | U8's pre-registered budget, stop on breach |
| δ_θ memorises the pool | The owner's rule: a success milestone. On-pool first, off-pool reported at the X8/X9 banks. The Smogon prior is never replaced, only ⊕'d |
| The bundle is two changes (tokens + flat pointer), so a strength regression cannot be attributed | The flat pointer needs the hypothesis candidates. α stays `detached`, so its effect on strength is only through the re-expressed cells. If NOT DETECTED, a cheap follow-up ablates the cell inputs to blob-equivalent, and another undetaches π (§3.2) |
| A mon with fewer than 4 moves breaks "mass 4" | The label shows it; it is an OTHER_move over-count. Report its rate; not designed for |
| An opponent team of fewer than 6 mons (possible off-pool, not in the pool) makes k = 6 − r over-count by the missing mons | Report its rate on any off-pool read; the presence mass is then too high by that count (not designed for) |
| 15M under-reads a late benefit | Named in §7.2; it falls on adoption, not safety |

### 8.2 Open gaps and findings (each a FINDING per standing rule 7)

| id | gap / finding |
|---|---|
| F-X5-1 | **Stale docs.** `ARCHITECTURE.md` §2.1 says 36 tokens and §2.3 says 29; production is 61 (32 event seats). `delivery_graph.py:973` counts 29. `design_q_head.md` §1 says "29 → 31" (pointer added here). `HiddenOppBeliefPool` reaches the critic too. Out of scope to fix in this doc unit (standing rule 9); reported. |
| F-X5-2 | The blob's physics harm (Jensen gap, attacker gate) is UNMEASURED. X5's justification rests on intent, calibration and Q / search, not on a measured physics loss. |
| F-X5-3 | **FIXED 2026-10-03 in `680edc36`** (`gen3_label_lookup_guard_v1`). Every Rust label writer returns an `Err` on a lookup it cannot make; 0 skips over 34,001 measured episodes. Pinned by `src/rust_env/tests/label_lookup_guard_test.rs`. |
| F-X5-4 | **FIXED 2026-10-03 in `0c25a1f4`** (`CHANGELOG.md` "F-X5-4"). The K9 harness built a different INIT at 8 threads (SB3's orthogonal re-init is a thread-count-dependent LAPACK QR: 15 of 41 groups moved). `build_learner` (and `rebuild-buffer`'s learner) now build at one thread and restore the caller's count; the banked golden is UNCHANGED. Pinned by `learner_golden_threads_test.py`. |
| F-X5-5 | **FIXED 2026-10-03 in `50fdfdc2`** (`CHANGELOG.md` "F-X5-5"). The production trainer's fresh build had F-X5-4's thread dependence (state_dict sha `eb409f05` at 1 thread vs `ff9276ac` at 8). `construct_fresh_learner` and the fresh fixtures now build inside the one shared `single_thread_build`. Pinned by `src/main/train/fresh_build_threads_test.py`. |
| F-X5-20 | Two different untaught opponents carry the 3.69 and 4.90 floors; they are not one scale. (This was "F-X5-5" in this note's first version; the CHANGELOG assigned F-X5-5 to the production thread pin, so it is renumbered here.) |
| F-X5-21 | **The T0 species prior's axis holds 13 phantom nums.** It is `[max_species = 400]`; only nums 1–386 hold a species, and `build_species_cooccur_prior` gives 387–399 the floor marginal 1e-4. A fixed-size construction over V = {nums ≥ 1} would carry them; U2 takes V = the dex-row table's `valid` mask minus the revealed nums (§8.3 U1 hand-off). Found in U1. |
| F-X5-6 | The sizing and battery non-inferiority rules used a CI that omits run variance (§7.1). |
| F-X5-7 | The anchors SOP's run floors at 100 games are mostly meter noise (SE ≈ 7 pp per cell). |
| F-X5-8 | `alpha_mask_rate` mixes non-choices with misses. No banked metric isolates the belief-miss share; U4 adds `opp_intent/other_label_rate`. |
| F-X5-9 | `UNDERSTANDING.md` l.586 and l.1346 still call the belief win-rate effect unmeasured; L21381 measured it. Fix in the next UNDERSTANDING pass. |
| F-X5-10 | The KL stop's controller, dose and `update_fit` couplings (§5.3), now X28's to handle. |
| F-X5-11 | **The production critic route's R1 `prov = Σα·w` = Σw²/Σw** (`pair_reduce.py:150-154`, forced on by `value_threat_inject`) breaks I2 although it sits in a renormalised mean. Classed M (§3.5). |
| F-X5-12 | **An undeclared `hp > 0` gate.** A hidden slot encodes HP exactly 0, so it is dropped from `p_pur_vs_us`, the v / t / g edges, the `pairwise_boost` speed gate and s1, beyond the declared `att_gate`. X5 rebuilds "alive" from `opp_addressable` (§3.4). |
| F-X5-13 | **The top-6 is computed twice under two tie rules** (`torch.topk` at `damage_op_blocks.py:875` and `:961`; `≥` in E5 at `pointer_head.py:147-150`). A candidate tied at rank 6 can fall in neither the seats nor the tail. Only `intent_axis_alignment_test` checks the two top-6 agree. X5 uses one ordering (§3.1). |
| F-X5-14 | `their_cls`, `value_cls` and HiddenOppBeliefPool use BOOL key-padding masks; a log-π bias needs FLOAT masks. Behaviour under compile UNVERIFIED; U3 checks first. |
| F-X5-15 | `intent_threshold`, `intent_move_cell` and `intent_conditional` do not apply `seat_live`; `pair_alpha` does (pre-existing). U4 makes them consistent. |
| F-X5-16 | No item, HP-type or spread label exists for unrevealed mons (`per_slot.rs:340-348`, `spread.rs:185-199`), so those aux heads are unsupervised on hypothesis seats. Moves can be supervised (`belief.rs:266-283`). |
| F-X5-17 | **Same seed ≠ same run on GPU.** A and A2 (seed 1001) diverged at the first update and ended 11.35 pp apart in U (§7.1). The previous "seed pairing" claim is withdrawn. |
| F-X5-18 | **No head-to-head run floor exists**, and the design's power at the borrowed σ = 3.43 is 0.57 (§7.4). |
| F-X5-19 | For a repeated dex number, species and moves labels take the FIRST truth mon while item, HP type and spread take the LAST (code survey; which label families production declares was not checked). UNVERIFIED as a defect; Species Clause makes it unreachable in a legal team. |
| F-X5-22 | **The unrolled 64-step bisection is the expensive part of the compile** (U2, MEASURED on CPU): ≈ 9 s per construction against ≈ 1 s with zero steps, superlinear in the step count (1.9 s at 8, 3.7 s at 32); three constructions per learner micro-step graph. If U8's GPU figure binds: run τ outside the compiled region (it is `no_grad`, so a graph break costs no gradient), or a `while_loop` / fewer fp32 steps (fp32 stops moving after ~25) — each a change to §3.2's "same count in every dtype", so an orchestrator decision. **GPU, MEASURED 2026-10-04 on `e78884c4` (U3 part 1; `designs/research_state/measurements/x5_u2_gpu_checks_2026-10-04/README.md`): the GPU compile cost does NOT bind.** Four production launches (two per arm): whole startup +35 s on ~410 s (T2 build 245–299 s from launch to launch under load for the SAME arm; R1 reset + prewarm 119.0 / 119.8 s blob vs 125.5 / 130.5 s fixed_mass); graphs unchanged (R1 4, T2 3), `recompiles_after_lock` 0, canary PASS. Isolated cold compile of the extractor (fwd+bwd, B = 2,048): blob 83.2 / 81.5 s, fixed_mass 94.3 / 97.0 s, the same graph at ZERO bisection steps 98.7 s, so the 64 unrolled steps cost ≈ 0 to compile on Triton and ≈ 0.4 ms of 83 ms to run. Measured as measurements only, not shipped: τ in an opaque eager op is SLOWER (+4.2 ms per micro-batch, +2.8 ms per T2 forward) with no compile gain; 28 steps gains nothing. The design stands. The early cost read at U3 part 1 is over the §3.6 budget for another reason: `train_ms` **+8.7 %**, in the token path's GEMMs (§3.6). |
| F-X5-23 | **OTHER's "next 32" tail mean (§3.3) adds a second selection boundary** that §3.1's rule did not name; it is now in the rule-8 exclusion (`near_tie_rows`). MEASURED 0.35 % of synthetic prior rows near-tied there (species sharing the 1e-4 floor tie exactly). A π-weighted mean over the WHOLE tail would remove that boundary at one `[B,S] @ [S,E]` matmul; not adopted (it changes §3.3), offered to U3. **RESOLVED in U3 (ORCHESTRATOR): the π-weighted mean over the WHOLE tail** — the boundary is gone, and it matches M3 (c)'s physics (`other_tail_probs`). |
| F-X5-24 | **The active's move presence π_m is a construction over MoveBelief's logits, but MoveBelief's loss is still its per-move sigmoid BCE.** §3.2 specifies the construction, not the move supervision; a set BCE on (a_m + τ_m) would mirror the species side but changes the shaping signal. **ORCHESTRATOR (U3): keep MoveBelief's per-move BCE; the move-side set BCE is a recorded follow-up, not built.** |
| F-X5-25 | **The census missed four sites and misclassified two** (U3 re-run, §3.5): Beat Up's party sum, `pair_alpha_full`, the α / β pointer heads; d4's top-K is class S, E5's `p_tail` class E. |
| F-X5-26 | **MoveBelief's reinjection soft-embed under fixed_mass still weights each move by its own sigmoid inclusion probability** (a presence-weighted sum, but not the fixed-mass π_m). A π_m reinjection at the active would also cut the PPO → move-head route there (M10's logic) — a semantics change left for the orchestrator (U3). |
| F-X5-27 | **`BeliefSlots` is built but never called under fixed_mass** (kept so both arms share every non-X5 parameter); its `unknown_slot_emb` gets no gradient in that arm. Deleted with the losing arm. |
| F-X5-28 | **OTHER's trunk seat has no edge cells** (the edge families write the 6-slot block) and the op prices no OTHER column yet: until U3 part 3, OTHER reaches the trunk as content + log-mass only. **CLOSED in U3 part 3** for D1 / C1 / C3 / D4 / V (the rest: F-X5-29). |
| F-X5-26 / 27 | **BUILT in U3 part 3 (ORCHESTRATOR).** F-X5-26: the active's reinjection reads the detached π_m (cost: no PPO → move-head gradient through that row; the BCE still trains the head). F-X5-27: `BeliefSlots` not built under fixed_mass (its init draw still runs, so non-X5 init bytes stay equal to blob's). |
| F-X5-29 | **OTHER has no edge in C2, S1, T, X (its opp cell) and G.** Their per-slot reads are TYPE / ABILITY ones (status immunity, trap / Levitate priors, Dark effectiveness, weather immunity, Early Bird), which need a tail expectation of each per-species table, not the stat expectation the built families use. An absent edge is bias 0 ("no information") — not a zero cell, which the map would turn into its learned bias. Hand-off item. |
| F-X5-30 | **The OTHER pass re-runs whole 6-column kernels to read one column** (D1, C1's five outgoing worlds + C1b, C3, D4, V). CPU compile of the fixed_mass extractor rose 100 s → 142 s (MEASURED, cold); the elementwise runtime cost is UNMEASURED (U8's budget). A one-column kernel variant is the fix if U8 binds. |
| F-X5-31 | **Two constructions price a hypothesis's bulk / speed.** D1 (and C1's outgoing worlds) use the expected-latent one-hot (the Smogon spread-prior means, SPECIES_EXP_MULT's expected ability immunity), as the brief specifies; V / C2's paralysis delta read the spread head's prediction at the hypothesis seat (unsupervised there, F-X5-16) and S1 / T / X / G the dex row's ability id, as for a revealed mon at first appearance. Consistent with the blob's revealed-vs-hidden split, but two numbers for one mon's speed. |
| F-X5-32 | **OTHER's attacker moves are parameter-free** (the E10 Smogon mixture over the tail), while hypotheses' are MoveBelief's learned rows: OTHER's threat cannot learn. Deliberate (M3 (c) = the blob's construction), recorded. |
| F-X5-33 | **The per-mon order ranks a revealed Hidden Power's 16 typed channels by P(t)**; tracker-narrowed equal P(t) can put a near-tie at a per-mon cut, so `near_tie_rows` may exclude more rows with a revealed HP (conservative; 0 on the golden buffer). The bench rows' move REINJECTION still uses sigmoid weights (F-X5-26 covers the active only). |
| F-X5-34 | **OTHER_move is a label only for one of its MEMBERS** (U4 decision). §3.7 says "a move outside the seats → OTHER_move"; as built, a move outside the seats that is not in the presence construction's candidate set (Struggle, a learnset gap — π = 0 in the model) is MASKED and counted (`flat_unmodeled_rate`; 0 on the K9 golden buffer, 0–4.0 % of choices across the fixed_mass smoke's five updates against bots), so OTHER_move's label means exactly what its mass means (Σ_beyond π) and its calibration reads by linearity. Same rule for OTHER_species (a hidden switch-in outside V's tail). **The composition of the unmodeled rows is UNVERIFIED** (candidates: Struggle, moves copied by Transform / Mimic, learnset-table gaps); U7 should break it down before it is read as a belief property. **U7 broke it down on the bank: F-X5-43** (all Struggle). |
| F-X5-35 | **`seat_live` consistency (F-X5-15) is applied under fixed_mass ONLY.** The three consumers take an optional `seat_live`; the blob arm passes none, so production stays byte-identical and keeps the pre-existing inconsistency until the losing arm is deleted. |
| F-X5-36 | **No human render of the flat pointer in traces.** `RLPlayer._opp_intent` and `main.search_dividend.alpha` read α / β only, so a fixed_mass trace carries no `opp_intent` block and the search-dividend α reader returns None. `flat_intent.render_flat` exists; wiring it into the player / prober is a follow-up (out of U4's scope). |
| F-X5-37 | **U4's added cost is UNMEASURED on the GPU** (§3.6 "What U4 ADDS"): OTHER_move's full-move-axis pricing (the status coordinates over `[B, 6, 400]`, a second `[B, 400]` damage-roll pair, the einsums). CPU compile 145 s cold vs 142 s (one run each). If U8's budget binds, the status coordinates can be computed on the tail's TYPE / category marginals instead of per move — a semantics change (orchestrator decision). |
| F-X5-38 | **The blob β's no-candidate rows read UNIFORM to its consumers** (pre-existing, blob only): `BetaSwitchHead` sets a row with no legal switch-in to all-zero logits (NaN-safety for the CE), so `has_cand` is true and `switch_branch` / the boom cell read a uniform arrival where there is none. The flat re-expression gives −inf there (zero switch mass). Reported, not fixed (standing rule 9; production byte-identity). |
| F-X5-39 | **The fixed_mass arm logs `opp_intent/flat_*`, not `alpha_*` / `beta_*` / `beta_setvalued_*`**; an arm-vs-arm intent read must map the keys (`flat_move_recall_top1` ↔ `alpha_move_recall_top1`, `flat_switch_tgt_top1` ↔ `beta_recall_top1` (named `flat_switch_target_recall_top1` before 2026-10-05, F-X5-48), …). |
| F-X5-40 | **Two degenerate rows, both finite and massless.** A MASKED OTHER_species' `out_cells` column reads the slot `other_col` falls back to (β puts −inf there); a row with NO live flat candidate (padding / an all-zero observation only) gets all-zero logits, so its re-expressed α / β are uniform to the consumers. Neither reaches a real decision. |
| F-X5-41 | **FIXED 2026-10-04** (`gen3_smogon_prior_denominator_v1`): the prior is `Moves / W`, W = `Σ Abilities` (Smogon's own `p.raw.weight`), Skarmory Spikes 0.997, per-species sums 4 less the empty-slot mass (min 3.79); throwing guards in the tool and at the facade load; both K9 golden arms re-recorded with proof (`research_state/measurements/move_prior_golden_2026-10-04/`); role set 15 → 29. WAS: **The Smogon move prior is DEFLATED** (found in U7; data-level, not fixed — a `data/` change is out of scope and forbidden beside a pinned live run). `tools/smogon_stats_downloader/compute_priors.py` writes P(m \| s) = chaos `Moves[m]` / `Raw count`, but the aggregated chaos `Moves` are RATING-WEIGHTED and `Raw count` is not: per-species sums are 0.41–3.75 (median 1.47) where a set runs ~4. Skarmory Spikes reads 0.547 against 0.997 normalised by Σ Moves / 4; Metagross Meteor Mash 0.571 vs 0.986; Blissey Softboiled 0.517 vs 0.975. It reaches every consumer of `build_move_prior_logits` (MoveBelief's prior, the E10 hidden-slot mixture, X5's OTHER and hypothesis moves at cold start) and the role set. MEASURED on the bank at cold start: every role's R1 Δ is negative except Surf / Substitute / Focus Punch, Hidden Power −1.10 carriers per team (M 1.52 vs N 2.63). The fix (normalise each species' moves by Σ Moves / 4) moves the role set and the prior column: a read after it must not be pooled with one before (`infer` refuses on `role_set_sha256`). |
| F-X5-42 | **`blob`'s α masks a Hidden Power click whose seat carries the WRONG TYPE** (pre-existing, blob only). The Rust label resolves a click to the TRUE typed num (`intent.rs`), the blob's seats are typed channels, so `match_seats_to_move_num` matches only when the seat's type is the true one. MEASURED on the bank (cold `blob`): of 735 Hidden Power labels, 625 rows hold a typed HP seat and only 584 the true type — 41 clicks masked in training although Hidden Power IS a seat (110 more have no HP seat at all: a genuine miss). U4 fixed it for `fixed_mass` (`hp_seat`: a typed label matches the revealed 237 seat; at U7's pre-U4 read, 418 of the 735 sat on a 237 seat). The reader collapses every Hidden Power into one event, so it is unaffected. Not fixed (production byte-identity; deleted with the losing arm). |
| F-X5-43 | **The composition of the unmodeled / missed rows on the bank (F-X5-34's request).** Training labels a Struggle a MOVE: 225 of the bank's 12,274 intent labels (1.8 %), all on stall-PP rows; neither α nor the flat pointer can name it (the flat fold masks it as unmodeled, α as a miss). Under the flat pointer at cold start NOTHING else is unmodeled on the bank (0 misses after Struggle). The `blob` α's misses (cold: 583 = 538 inside the active's learnset — moves outside its six seats — and 45 outside it) are seat-coverage misses. The reader excludes Struggle (forced, no PP — not a choice) and counts it. |
| F-X5-44 | **FIXED in U6 (byte-identical).** OTHER's averaged move presence (`other_roster`) ran its fixed-size construction and its stable ORDER over every legal move even on a row whose OTHER is DEAD (every opponent species known: `P_tail` = 0, so the mixture is UNIFORM). The uniform π's exact ties were counted by the declared `sort_head` site as rule-8 near-ties: 29 of the golden buffer's 64 rows, a K9(b) excluded share of 37.5 % on a 16-row micro-batch against the 0.15 ceiling (FATAL). The order is read only through `where(hyp, …)` and the Pursuit presence only times `other_any` = 0 there, so the candidates are now masked by `other_live`. The six hashes (state_dict, features, logits, values, log-probs, gradients) are unchanged in both arms; the excluded share went from 30 / 64 to 1 / 64. Reverting it fails `learner_golden_fixed_mass_test`. |
| F-X5-45 | **A golden buffer's rebuild is not bit-reproducible in its behaviour columns.** Rebuilding the blob buffer today replays the SAME games (obs, actions, masks and rewards byte-identical), but `values` / `log_probs` / `advantages` / `returns` differ by ≤ 7.2e-7. That is rounding in the collector's T2 forward, pre-existing and not X5's. A `rebuild-buffer` therefore moves a golden even with no code change, so re-record after every rebuild (the harness already requires it). The fixed_mass buffer rebuilt byte-identically twice in one session. |
| F-X5-46 | **FIXED 2026-10-04** (the F-X5-41 re-record wrote blob's `init_group_sha256`, 41 groups). WAS: **The blob entry has no `init_group_sha256`** (it was recorded before U6, and U6 left it untouched). `diff` compares the field only where it is recorded, and the next deliberate blob re-record adds it automatically. Until then, a blob INIT move is named only by the whole-model hash. |
| F-X5-47 | **FIXED 2026-10-04** (`gen3_smogon_species_usage_weighted_v1`): `species_usage()` is each species' W (`Σ Abilities`, the facade's `_weighted_count`), checked at build against `Σ Abilities` and `Σ Moves / 4` (`PriorInvariantError` on a Raw-count table); new / old share ×0.19–×1.75 over all 216, ×0.92–×1.19 over the top 25 (Tyranitar 0.078 → 0.086, Shuckle ×0.19); both K9 golden arms re-recorded on rebuilt buffers holding the SAME games, with proof (`research_state/measurements/species_usage_golden_2026-10-04/`); role set 29 → 28 (Ice Punch drops). WAS: **The species USAGE marginal is UNWEIGHTED** (F-X5-41's class, found in its fix; not changed — out of that unit's scope). `gen3_data.priors.species_usage()` reads the chaos `Raw count` (every rating, weight 1), while every other Smogon prior is rating-weighted. It feeds `build_species_usage_prior`: the op's `SPECIES_USAGE_PRIOR`, the T0 marginal, and the `belief_roles` usage. The co-occurrence LIFT `log P(s \| t) / P(s)` divides a WEIGHTED teammate conditional by it. On the 2025-05 .. 2026-04 window, log(weighted share / raw share) runs −0.08 .. +0.14 over the top-25 species (Tyranitar +0.10) and −1.65 .. +0.56 over all 216 (Shuckle −1.65). The weighted count is `Σ Abilities`, which agrees with Smogon's latest-month weighted `usage` within ~5 % for the top species. A fix moves the T0 prior, the op and the role set, so it needs its own golden proof. |
| F-X5-48 | **FIXED 2026-10-05** (`gen3_fm_index_max_v1`, `research_state/measurements/x5_fxc4_nanfix_2026-10-05/`). Two launch blockers of the fixed_mass arm. (a) **The compiled R1 backward was NaN on CUDA** (F-XC-4: 41 parameters, 8 of 8 compiles): Inductor recomputed the incoming direction's 400-wide sweep inside the backward of its ten `amax` channel maxima, and Triton's FMA contraction rounded the recompute differently from the forward kernel, so on some rows no element equalled the saved max and `amax`'s backward divided by a tie count of 0. Under fixed_mass the ten maxima are now selected by index (`damage_op.max_by_index`); value bit-identical, blob untouched (byte-identical R1 / T2 compiled code). (b) **A fixed_mass self-play run crashed at its first logger dump** once two opponent classes held rows: the stdout table (36 characters) truncated the per-class `flat_switch_target_recall_top1_{bot,pool,stable,exploiter}` to one key. Renamed `flat_switch_tgt_top1` (≤ 33 characters with any class suffix); the TensorBoard tag `opp_intent/flat_switch_target_recall_top1*` is DISCONTINUOUS at this commit (only the cost-ablation and F-XC-5 runs wrote it). |
| G-1 | σ_run at 15M, the H2H run floor and snapshot jitter are unmeasured (P0 measures the last two). |
| G-2 | Negative evidence (an opponent NOT switching to X) is not modelled; X12. |
| G-3 | OTHER's embedding: `design_q_head.md` §10, logged by U8 (OTHER's attention share). The budget question is now §9 M3. |
| G-4 | That the Lane S bank exposes both full teams to a reader: VERIFIED (Tier 0 F1; U7 reads them, 0 label mismatches). |
| G-5 | PokaiTrainer (arXiv:2608.29197): its belief representation is unread. Read it before X4 returns. |
| G-6 | Whether the op's per-hypothesis attacker rows fit the elementwise budget: unmeasured (U8). |
| G-7 | The live head's on-pool sharpness, hence X5's real OTHER mass on-pool: UNVERIFIED (§3.3; a species-only proxy is a floor on sharpness). |
| G-8 | The CPU cost of one H2H cell (2,000 games, checkpoint against checkpoint): unmeasured (U0). |

### 8.3 Build plan (after the deletion pass, before X26)

Sizes are in agent-days. A "tier" is the gate a unit must pass before it lands. **Every unit is Opus-tier.**

| unit | what | size | tier / gates | agent |
|---|---|---|---|---|
| U0 | **DONE 2026-10-04 (F-U6-1 CLOSED the same day).** The checkpoint-vs-checkpoint mirrored H2H CROSS CLI (`sprt.py` + Rust eval core; the A/B's primary meter) + the P0 planning reads (§7.3) + its per-cell CPU cost. AS BUILT: `main.h2h play` (P0, `54b78aed`; ledger v2 `e5f393cb`) + the multi-cell engine `main.h2h play-many --players <X5 seeds> --opponents <blob seeds> --purpose ab --family <A/B> --request <look>` (eval U6: one engine, weights loaded per cell, games byte-identical to single-cell `play`, `measurements/h2h_multicell_2026-10-04/`). **The cross's TWO architectures play on ONE engine** (`main/h2h/arch.py`): one T2 slot group per arm, each with only the slot its side needs (X5 = the player's eval slot, blob = the opponent's sentinel slot), one eval core per (player arm, opponent arm); a same-architecture cell plays the single-group games on it, a cross cell replays exactly (`measurements/h2h_cross_2026-10-04/`, `play_cross_integration_test.py`). GPU memory ESTIMATED (≈ today's engine), the GPU start + memory DEFERRED to the GPU owner (commands in that README). The h2h PROTOCOL is unchanged (`gen3_eval_protocol_v1_h2h`) | 1 | targeted + static; CPU under `mem_cap.sh` | opus-high |
| U1 | **DONE 2026-10-03** (`gen3_x5_dex_rows_v1`; `CHANGELOG.md` "X5 U1"). Dex-row table generator (Rust encoder) + committed artifact + `sim`-tier byte gate + the real-state cross-check (§3.4). (The `belief.rs` guard is DONE, `680edc36`.) Hand-off below | 1 | `sim` + cargo + static | opus-high |
| U2 | **DONE 2026-10-03** (`gen3_x5_hypothesis_set_v1`, config v136; `CHANGELOG.md` "X5 U2"). T0 hypothesis builder: δ_θ, the fixed-size construction (bisection, structural k = 0 / k = n, logsumexp OTHER), the single stable ordering, set BCE, BeliefHead re-target, moves; the `--belief-tokens` flag, versioning, registry; compile-time cost of the unrolled bisection (§3.6). Hand-off below | 2.5 | routine gate; tier contract; flag gates; the construction's tests | opus-high |
| U3 | **DONE 2026-10-04** (`gen3_x5_belief_tokens_v1`; part 3 = the op's opponent-MON axis + OTHER's physics, `CHANGELOG.md` "X5 U3 part 3", the U3 part-3 hand-off below). Parts 1–2 (`be6ba590`, `5697c762`; `CHANGELOG.md` "X5 U3 part 1 / part 2"): census re-run (§3.5); hypothesis tokens, OTHER's trunk seat, log-π bias in the trunk and the four class-E pools (float masks, CPU compile checked), species-specific T0 heads, MoveBelief's hypothesis-seat rule; the active's move axis (one order, the presence-scaled max, the revealed-HP seat, OTHER_move). **Part 3 NOT built** — the op's opponent-MON axis (hypothesis defenders / attackers, "alive" from `opp_addressable`, OTHER's physics per M3 (c), class M over mons): see the U3 hand-off. Tokens into the chain: re-run the 41-site census on the built code; log-π bias in the transformer and every class-E pool (float masks, compile check first); class-M semantics per §9 M2; the op with hypothesis defenders and attackers, "alive" from `opp_addressable`; OTHER's physics per §9 M3; E5 owner bias; aux heads on hypothesis seats (§3.4); I1 / I2 and class-M / class-S tests per site | 3 | routine gate; invariance tests; obs golden untouched | opus-xhigh (GIGO risk; the orchestrator dispatches it) |
| U4 | **DONE 2026-10-04** (`gen3_x5_flat_pointer_v1`; `CHANGELOG.md` "X5 U4"; the U4 hand-off below; §3.7 "As built (U4)"). Flat α pointer + OTHER labels; re-expressed cells with OTHER priced and `seat_live` consistent (§3.7); B ride-along re-base; `other_label_rate` | 2 | routine gate; `ridealong_update_test` bit-identity | opus-high |
| U6 | **DONE 2026-10-04** (§6 "As built (U6)"; `CHANGELOG.md` "X5 U6"; the U6 hand-off below). K9 golden: `init_group_sha256`, the second entry (`arms.fixed_mass`, its own buffer, name-keyed noise), fp64 references, teeth tests, the δ_θ / B isolation at golden level, K9(b) passing (excluded 6.25 %). F-X5-44 fixed, byte-identical. (The thread pins are DONE: harness `0c25a1f4`, production fresh build `50fdfdc2`.) | 0.75 | routine gate | opus-high |
| U7 | **DONE 2026-10-04** (`CHANGELOG.md` "X5 U7"; the U7 hand-off below). Readers: `main.belief_roles` (R1–R4) + the intent / presence / OTHER purpose reads on the Lane S bank, per-run values for §7.4's across-seed inference (`infer`); `fixed_mass` read through U4's flat pointer | 1.25 | targeted + static | opus-high |
| U8 | Smoke + the first-two-minutes real launch + GPU cost budget (§3.6) via `gpu_lock`; build thread count recorded | 0.75 | `--debug` smoke, real launch, learner benchmark | opus-high |
| — | Full suite before ship (`pytest src/ -q`), slow tier on the X5 commit | — | before `/gen3ai-ship` of U2–U6 | — |
| A/B | 6 → 10 → 16 runs via the training agent | GPU 15.3 → 25.5 → 40.8 h (expected ≈ 26–34) | §7.4, committed before the first game | training agent |

**U5 (the KL early stop) is DROPPED** (owner: OFF). **Total about 11–13 agent-days + up to 41 GPU-h** (12.25 at the
point sizes; +0.5 if the owner picks M3 option (a), +0.5 if M2 picks option A). The backlog's "≈ 2 more attention
tokens and a small model-version bump" understated the build: the physics, pointer and the reduction census are the
work.

**Order.**
- After the deletion pass; U0 can run now (CPU).
- U1 → U2 → U3 → U4 is the chain. U6 can run in parallel with U2–U4. U7 runs in parallel; U8 last.
- U3 cannot start until the owner answers §9 M2 and M3.
- No Rust core runtime change, so M5's parity gates are a re-run, not a rewrite.
- The registration of §7.4, with P0's σ̂_h in its power line, is committed BEFORE any A/B game.

**U1 hand-off (read before U2).**
- **What exists.** `agents.model.hypothesis_dex_rows.load_hypothesis_dex_rows(layout["max_species"])` returns
  `rows` `[400, 122]` float32 (READ-ONLY; take `.copy()` before `torch.from_numpy`, and register it as a
  NON-persistent buffer: data-derived, never a saved weight), `valid` `[400]` bool, `species` (the id per num) and
  `cells` (the declared cell classes) plus the table's `sha256`. Row index = national-dex num, the T0 prior's own
  axis. The artifact is `src/agents/model/hypothesis_dex_rows.json` (386 rows, 263 KB); regenerate with
  `python -m agents.model.hypothesis_dex_rows --write` when the `sim` byte gate says it is stale, and review the diff.
- **V must be `valid` minus the revealed nums, structurally.** The T0 prior is `[400]` but only nums 1–386 hold a
  species; `build_species_cooccur_prior` gives nums 387–399 the floor marginal 1e-4 like any rare species, and they
  have no dex row. §3.2's "V = {species numbers ≥ 1}" read literally would put 13 nonexistent species into the
  fixed-size construction (finding F-X5-21). With V = valid minus revealed, n = 386 − r.
- **What a hypothesis row says.** It reads as a POPULATED mon of a known species: `species_known` = 1, HP fraction
  1.0, recency saturated at 1.0 (never seen), protect odds 1.0, no item, no move, empty Hidden-Power block, not
  active. A one-ability species carries that ability as KNOWN (`[num, 0, 1, 1]`, poke-env's inference, as the encoder
  does for a real one); a two-ability species carries the Smogon top-2 prior. Today's hidden slot encodes HP 0 and
  `species_known` 0, so nothing that reads those cells may treat a hypothesis as revealed: the `hypothesis_marker`
  (§3.4) and "alive" from `opp_addressable` (F-X5-12) are what tell them apart.
- **Formes.** One row per base form. Castform (351) enters in weather as a weather forme (Forecast), so its row (Normal
  type) is the pre-entry hypothesis only; the Deoxys formes are Ubers; the Unown letters share stats and types.
- **No dependence on the fixed-damage GIGO fix** running in parallel (`damage_op_blocks.py` / `MOVE_BP`): a hypothesis
  row has no move cells and no damage-op column. A `data/pokemon` change that moves a species, ability-prior or item
  row DOES move it, and the byte gate says so.
- **Pin hazard.** The table is pinned with the code, but a pinned run's Rust encoder reads `data/` from MAIN at
  runtime; a `data/pokemon` change while a pinned X5 run is live would let real rows and table rows drift apart
  (standing rule 6 already forbids that change).

**U2 hand-off (read before U3).**
- **What exists.** `agents/model/hypothesis_set.py`. Under `--belief-tokens fixed_mass` the extractor builds
  `HypothesisBuilder` (T0, after `MoveBelief`) and stashes one `HypothesisSet` per forward as `fe.last_hypothesis`;
  NOTHING in the policy / value forward reads it yet (pinned bit-for-bit: `hypothesis_set_test`). `blob` builds
  nothing. Fields U3 consumes:
  - `species` (a `Presence`: `pi`, `log_pi`, `cand`, `k`, `n`, `live`, `full`; `logits` is the ONLY graph-carrying
    field and is for the presence BCE alone — never read it in the forward, M10);
  - the ONE order (`order`, `rank`) and the hypotheses: rank j's num `hyp_species[:, j]`, live iff j < k;
  - **seat placement (decided here):** hypothesis rank j sits in the j-th HIDDEN opponent slot in slot order —
    `slot_species`, `slot_log_pi` (0 on a revealed slot: the key bias U3 adds), `slot_rows` (U1's dex row, 0 on a
    revealed slot), `slot_is_hypothesis` (= the believed mask);
  - OTHER: `other_mass`, `other_log_mass` (`MASKED_LOG_PRESENCE` = −1e9 when masked), `other_live`, `other_tail_mean`,
    `other_token` (the learned rest vector + `other_map`; these two parameters get NO gradient until U3 reads the
    token);
  - the active's move group `moves` (`MovePresence`): `seat_nums` / `seat_live` / `seat_revealed` / `seat_pi` (the
    `K = entity_topk_seats` seats, revealed moves first by num, then the top unrevealed by π_m), `other_mass` /
    `other_log_mass` / `other_live` (OTHER_move, for the active's E5 `p_tail`), `tie_gap`.
- **Not built (U3's):** `hypothesis_marker`; `pokemon_encoder` on `slot_rows` (the encoder reads the whole `ctx`, so
  substituting hidden rows must keep the REAL `opp_believed_mask` / `opp_addressable`, never re-derive them from a
  hypothesis row's species-known 1 / HP 1.0); E4 / E5 re-wired to `moves.seat_nums` (retiring both `torch.topk`
  calls, F-X5-13); a constructor refusal of `entity_topk_seats < 4` (a fourth revealed move would have no seat; U2
  dropped that check because the registry cannot express a per-value requirement).
- **Revealed Hidden Power** occupies a move seat as the typeless num 237; its 16 typed channels are removed from the
  candidates (they are that revealed move's type distribution). U3 must render the seat through `compose_typed_hp`'s
  typed weights.
- **Rule 8.** `near_tie_rows(hs)` is the exclusion for every two-computation check (seats, OTHER's tail-mean cutoff,
  move seats), at `SELECTION_TIE_EPS` = 1e-6. 🚨 `selection_sites.py` declares the builder's three float-operand
  discrete ops `NOT_LOGP` — true only while U2's stash is unread. **U3 must re-declare them** when log π starts
  reading the set: the argsort as a MARGIN rule on the boundary gaps; the bisection's `total > k_t` needs a reason of
  its own (it is not a discontinuity: τ converges to the same root either way); `denom > 0` is a count gate.
- **Init isolation.** Every X5 module is built inside `fork_rng(HYPOTHESIS_INIT_SEED)` from `IsolatedLinear` (SB3's
  orthogonal re-init skips it), so `fixed_mass` leaves every non-X5 initial byte equal to `blob`'s (pinned on an
  unperturbed real build). Build U3's / U4's modules the same way. The K9 harness's `perturb_` draws noise per
  parameter in `named_parameters` order, so any PERTURBED comparison of non-X5 groups needs name-keyed noise (U6).
- **Supervision as built.** The `hidden_team_set` belief-bank row (gated EXCLUSIVELY against `hidden_team`, same
  coefficient `opp_belief_aux_coef`): presence BCE (δ_θ's only gradient) + BeliefHead's set BCE (its per-slot
  species logits reduced to the hidden-slot mean, then the same construction) + `moves_weight` × BeliefHead's moves
  BCE on hypothesis seats, supervised iff the seat's species IS on the true unseen team. A row whose labels do not
  count k is dropped and reported (`belief/set_presence_label_mismatch`; 0 on the golden buffer and the smoke). In
  this arm BeliefHead's per-slot species logits are no longer supervised per slot, so β's content-addressed target
  degrades until U4 retires β. MoveBelief's own per-move BCE is UNCHANGED; whether the active's π_m should get a set
  BCE is open (F-X5-24).
- **Compile cost** (§3.6, MEASURED on CPU): ≈ 9 s per unrolled construction, ≈ 18 s for the builder. GPU, MEASURED
  2026-10-04 (F-X5-22): the unrolled steps cost ≈ 0 on the GPU, so the options F-X5-22 names are not needed.
- **Smoke.** `--debug --steps 10000 --arch production --belief-tokens fixed_mass --allow-nonproduction-arch` does not
  reach an update (the production recipe's 98,304-row target); add `--n-envs 1 --batch-size 384
  --rollout-target-samples 2304 --grad-accum-steps 1 --n-epochs 2` (exit 0 at U2: hypothesis recall 0.42, OTHER share
  0.61 — §3.3's cold-start range).

**U3 hand-off (parts 1–2; part 3 is now BUILT — read the part-3 hand-off after it).** U3 landed in two code parts
(`gen3_x5_belief_tokens_v1`; `CHANGELOG.md` "X5 U3 part 1" / "part 2"). Production stays `blob`,
byte-identical (six hashes on the K9 golden buffer — state_dict, (pi, vf) features, action logits,
values, one backward's gradients — equal `7b46d62f`'s after each part). What exists, and what does not:

- **Built (part 1, class E).** `agents/model/hypothesis_tokens.py`. The builder runs in two halves:
  `species_set` at T0 BEFORE `MoveBelief`, `with_moves` after it. `hypothesis_ctx(ctx, hs, layout)` is
  the context with each hidden opponent slot's row = its dex row and the ids re-sliced (every mask the
  REAL one); THE `PokemonEncoder` runs a second pass on it and the hidden slots take that token +
  `hypothesis_marker` (`splice_hypothesis_tokens`). The T0 move / HP-type / item / spread heads read
  hypothesis seats with their species (`_apply_move_belief(..., hctx)`, `_spread_hp_damage(..., hctx,
  hs, fm)`). OTHER_species is ONE extra trunk seat right after the entity seats (index
  `team_transformer._total_tokens + entity_seats.n_seats`; the event seats stay last), typed
  THEIR_TEAM. The key log-presence (`key_log_presence`) rides the trunk bias; `OppPresence` feeds
  `their_cls`, `value_cls`, `HiddenOppBeliefPool` and `value_entity_pool` (OTHER joins each). The
  forward's local `_presence` / `_hs` / `_hctx` / `_fm` are the handles U4 reads.
- **Built (part 2, the active's move axis).** `FixedMassMoves` (`fixed_mass_moves(hs.moves,
  typed_logits_at_active)`): the ONE order's seats feed the E4 seats (`entity_seats.last_cand =
  (seat_nums, seat_w)`), the op's seat axis (`last_topk_idx` = seat_nums, `last_topk_w` = seat_w,
  `last_pair_seat_live` = seat_on, plus `stash.seat_ext_idx` / `stash.seat_mix`), α's seats
  (`alpha_seat_nums`, `seat_valid = seat_on`), D3 / S3 and the intent operands. **Any new seat-axis
  consumer that keys physics by move NUM must run on `idx_ext` and contract with `mix_seats`** — a
  revealed Hidden Power's seat is num 237 (BP 0, typeless) and is priced only through that mixture.
- **NOT built — U3 part 3, the op's opponent-MON axis** (the census rows below are its checklist; it
  is the GIGO-heaviest part and should be dispatched as its own unit):
  1. *Hypothesis DEFENDERS.* `_outgoing_matrix` / d1 / c1 (`pairwise_outgoing`, `pairwise_boost`) price
     a hidden slot through `unrevealed_species_probs(ctx, species_probs)` — today the T0 marginal,
     `[B,S]`, shared by every hidden slot, forced alive, P(KO) NULLED. The op already accepts a
     per-slot `[B,6,S]` override: pass one-hot(`hs.slot_species`) at hypothesis slots (E[stats] over a
     one-hot IS the concrete defender), and decide P(KO) (a hypothesis is pristine at full HP, so it is
     defined — un-null it, and say so). `opp_p_ghost` follows from the same per-slot belief.
  2. *Hypothesis ATTACKERS.* `att_gate = (1 − believed)·(hp > 0)` (`damage_op_pairwise.py:303-304`;
     c1b / c2 / c3) and d4's `revealed·alive·not_active` (`:868-872`) open for hypotheses, with
     "alive" from `ctx.opp_addressable` (F-X5-12), the attacker's stats from the hypothesis species
     (read `hctx`'s species / types, never its HP cell to decide), and its move weights from its own
     (now species-specific) MoveBelief row through the fixed-size construction at k = 4 (§3.2: "A
     hypothesis seat's own move posterior … uses the same construction at k = 4"). Each attacker's
     row is then scaled by its presence π_j at the max (class M, §9 M2 = C). `p_pur_vs_us` (`:1055-1057`)
     likewise. The active's row in `_believed_attackers` still top-Ks by sigmoid: give it the ONE order.
  3. *The other `hp > 0` gates on opponent slots* (F-X5-12, census): v (`:827-830`), t (`:1087-1088`),
     g (`:974-980`), the `pairwise_boost` speed gate (`:254-256`), s1 (`damage_op_blocks.py:1190-1229`),
     x's `opp_cells` (`:1068`) — "alive" from `opp_addressable` under fixed_mass.
  4. *Beat Up's party sum* (`damage_kinds.beatup_party_opp`, class E linear, MISSED by the first census):
     under fixed_mass its exact expectation is Σ_s π_s · baseAtk(s) over EVERY candidate (hypotheses +
     tail) — i.e. the team-level belief `π / k`. Use the BOUNDED gather form when touching Beat Up
     (`174f5e62`: never build an index as a sum of flags that can overlap).
  5. *OTHER's physics (M3 (c), ORCHESTRATOR F4 (a)).* OTHER = today's AVERAGED defender AND attacker
     on the renormalised tail `hs.other_tail_probs` (`P_tail @ tables`, the blob's own construction;
     its stats are the tail's EXPECTED stats). It needs a 7th opponent column wherever OTHER is read
     (OTHER's trunk seat currently gets NO edge cells — the edge families write `opp = slice(6, 12)`;
     the β / out_cells consumers are U4's), and it must NEVER read `e_mult = 0` (IMMUNE). At a
     max-type site OTHER contributes with presence `hs.other_any` = 1 − Π_tail(1 − π) ∈ [0, 1]
     (ORCHESTRATOR F4 (b); computed and stored, not yet read — Tier 0 capped the mass at 1 instead).
     Tests to write: OTHER priced from the tail (never immune), its max-site presence in [0, 1] and
     equal to 1 − Π(1 − π), the hypothesis attacker gate open, class-M I1 + regression on each site.
- **Decisions in force.** π DETACHED everywhere it weights the policy / critic (M10; the U2 + U3
  gradient tests pin it). OTHER averages the WHOLE tail (F-X5-23). MoveBelief keeps its per-move BCE
  (F-X5-24; the move-side set BCE is a follow-up). The bisection's compile cost is U8's (F-X5-22). No
  `ARCH_SIGNATURE` bump (§3.8: the version floor and the signature are DECOUPLED at adoption — pending
  the legacy manifest's D-L1).
- **Two semantics left as they are (FINDINGS for the orchestrator).** (a) MoveBelief's reinjection
  soft-embed still weights each move by its own sigmoid inclusion probability; a π_m reinjection at
  the active would also cut the PPO → move-head route there. (b) `BeliefSlots` is built but not
  called under fixed_mass (both arms share every non-X5 parameter); its parameter gets no gradient.
- **U4 starts from:** the forward's `_hs` (species + moves), `_fm` (the move axis, the seat stash),
  `_presence` (OTHER's refined token at `_seat_out[:, _other_idx − _total_tokens]`), and the op's
  `stash.seat_mix`. The α / β consumers (`pair_alpha`, `intent_threshold`, `intent_move_cell`,
  `intent_conditional`, `switch_branch`, `conditional_threat`) are untouched and still read α over the
  K seats + SWITCH and β over the 6 slots.
- **Smoke.** U2's recipe (`--debug --steps 10000 --arch production --belief-tokens fixed_mass
  --allow-nonproduction-arch --n-envs 1 --batch-size 384 --rollout-target-samples 2304
  --grad-accum-steps 1 --n-epochs 2`): exit 0 after each part; hypothesis recall 0.39–0.43, OTHER
  share 0.61–0.67.

**U3 part-3 hand-off (read before U4).** Built in `gen3_x5_belief_tokens_v1` (`CHANGELOG.md` "X5 U3 part 3");
production stays `blob`, byte-identical (the six K9-golden-buffer hashes equal `ecf9eeca`'s).

- **What exists.** `hypothesis_tokens.OpRoster`, built per forward by `build_op_roster` (after the move belief) and
  `other_roster`, stashed on the op as `op.stash.x5` (passed to `DamageOperator.forward` as `x5_roster`; the op and every
  opponent-axis kernel then run on the HYPOTHESIS context `_opctx`, the defender belief `_sp` = the per-slot one-hots).
  Fields U4 reads: `alive` (= `opp_addressable`), `hyp`, `slot_pi`, `species_probs`, `team_probs` (π / k), `move_w` /
  `move_order` / `move_rank` (per-mon fixed-mass move presence and its one order), `other_live`, `other_any`,
  `other_col`, `other_pursuit`, and `other` — the OTHER-MODE roster (every hidden slot = the tail-averaged mon:
  `att_base`, `has_type`, `spe`, `spe_std`, `override`; `concrete` = 0 on its slots, so P(KO) stays nulled).
- **OTHER's cells.** `ExtractorForward._other_edge_cells(_opctx, _x5r, sb, fams)` runs the op kernels under the
  OTHER-mode roster and reads OTHER's column (`other_column`); `EdgeBias` writes D1 / C1 / C3 / D4 / V at the OTHER seat
  (`EdgeBias.OTHER_FAMILIES`, zero when OTHER is masked). **U4's downstream cells** (`switch_branch`,
  `intent_conditional`, `conditional_threat`, `pair_alpha` …) read `op.stash.out_cells` / `opp_p_ghost`, which are
  STILL six columns: give them OTHER's 7th column from the same OTHER-mode D1 pass (`out_cells`) and
  `other_tail_probs @ SPECIES_IS_GHOST` (P(Ghost | OTHER), linear and exact) — never a zero row (`e_mult` 0 = IMMUNE).
- **Kernel contract.** A new opponent-axis kernel branches on `self.stash.x5`: alive from `roster.alive`, candidates
  from `roster.move_order`, tail tables through `_x5_avg` / `_x5_stab`; add a row to `x5_opp_mon_axis_test.py`'s
  "no hidden mon dropped" test, and to `OTHER_FAMILIES` + `_other_edge_cells` if OTHER should see it.
- **Decisions in force.** P(KO) un-nulled for hypotheses, nulled for OTHER (§3.4 as built). Per-(seat, mon) cells are
  not π-scaled (the key bias carries presence); only the max over mons is. OTHER enters a max site with `other_any`,
  never its mass. F-X5-26 / F-X5-27 BUILT. No `ARCH_SIGNATURE` bump (D-L1 pending, decoupled).
- **Open (findings).** F-X5-29 (OTHER edges for C2 / S1 / T / X / G), F-X5-30 (the OTHER pass's cost — U8), F-X5-31
  (two bulk / speed constructions for a hypothesis), F-X5-32 (OTHER's moves are parameter-free), F-X5-33 (revealed-HP
  typed channels at a per-mon cut; bench reinjection still sigmoid).
- **Measured.** CPU Inductor compile of the fixed_mass extractor = eager to 4.8e-6 / 2.2e-6 on 64 golden rows (0
  near-tie rows), 142 s cold. GPU compile and runtime: DEFERRED (U8). The Tier-0-style sanity read and the 15 tests
  are in the CHANGELOG entry.
- **Smoke.** U2's recipe, unchanged.

**U4 hand-off (read before U6 / U7 / U8).** Built in `gen3_x5_flat_pointer_v1` (`CHANGELOG.md` "X5 U4"); production
stays `blob`, byte-identical (the six K9-golden-buffer hashes — state_dict, pi / vf features, logits, values, one
backward's gradients — equal `889add9d`'s). No config bump, no `ARCH_SIGNATURE` bump (D-L1 pending, decoupled).

- **What exists.** `agents/model/flat_intent.py`: `FlatIntentHead` on the extractor as `flat_intent_head` (T2;
  fixed_mass only), `fe.last_flat_intent_logits` [B, K+8] (the publication; `belief_supervision("flat_intent_logits")`
  is the LIVE view), `fe.last_flat_intent` (`FlatIntentInputs`: `live`, `cand_ids`, `log_pi` and the label-side
  tensors) and `fe.stash.flat_consumer_ops` (`FlatConsumerOps`: the consumers' α / β and OTHER-extended operands).
  Column helpers `flat_width` / `other_move_col` / `slot_col` / `other_species_col`. Labels ONLY through
  `flat_intent_targets` (it returns the label class too: `LABEL_*`). α / β are retired in this arm
  (`last_alpha_logits` / `last_beta_logits` are None there).
- **For U6 (the K9 fixed_mass golden).** New parameter groups: `features_extractor.flat_intent_head.{hidden,out}`;
  α / β's groups are ABSENT in that arm (retired after the ortho draws — the non-X5 init bytes still equal blob's). New
  loss key in the fold: `opp_intent/flat_loss` (+ `other_label_rate`); `beta_setvalued_*` are not emitted. The flat
  fold is pure (no RNG). Coverage on the 64-row golden buffer: OTHER_move live on 62 rows, OTHER_species on 35; at
  cold start (aligned labels, one `train()`) `other_label_rate` 0.19, `flat_unmodeled_rate` 0.
- **For U7 (readers).** `other_label_rate` is the belief-miss share of CHOICES; the OTHER labels make the pointer's
  calibration on OTHER readable (P(OTHER_species) vs its label rate). `render_flat` names the options for a trace.
  `main.ridealong_read` passes the flat fields to B (`RideAlongSpec.opp_flat_k`).
- **For U8 (cost).** §3.6 "What U4 ADDS" (F-X5-37). The OTHER_move column is the one full-move-axis computation U4 adds;
  the flat head replaces α + β (24,833 vs 65,923 parameters).
- **Decisions in force.** π detached into the pointer (M10; the flat loss reaches the head, never δ_θ —
  `flat_intent_test`). OTHER_move / OTHER_species labels only for members (F-X5-34). `seat_live` in the three
  consumers under fixed_mass only (F-X5-35). The set-valued partial credit is superseded in that arm.
- **Open (findings).** F-X5-36 (no trace render), F-X5-37 (GPU cost), F-X5-38 (blob β's uniform dead rows, pre-existing),
  F-X5-39 (metric key mapping across arms), F-X5-40 (two degenerate, massless rows).
- **Smoke.** U2's recipe, unchanged (`--debug … --belief-tokens fixed_mass --allow-nonproduction-arch --n-envs 1
  --batch-size 384 --rollout-target-samples 2304 --grad-accum-steps 1 --n-epochs 2`).

**U7 hand-off (read before U8 and the §7.4 registration).**
- **What exists.** `python -m main.belief_roles read --out <dir> --ckpt <run>/…/<ckpt>.zip=<label> …` reads any checkpoint
  of either arm on the Lane S bank (re-encode 32–44 s once per invocation, then 9–15 s of CPU forwards per checkpoint at
  4 threads, beside a GPU job; peak 5.1 GB for four) and writes one JSON per checkpoint: `per_run` (the on-pool values
  §7.4 infers on), `on_pool` / `off_pool` blocks (intent all / move / switch / by opponent class + the miss breakdown;
  for the arm AND the prior column: presence, OTHER, roles R1–R2, R3), every rule-8 and Struggle exclusion counted.
  `python -m main.belief_roles infer --treat <X5 JSONs> --control <blob JSONs> --boundary <t> [--metric …]` is the
  across-seed two-sample t (pooled variance, df = n_t + n_c − 2, crossed iff t − boundary ≥ 1e-9).
- **The intent read under each arm.** `blob`: α / β (`forward._event_logp`); `fixed_mass`: U4's flat pointer
  (`forward._flat_event_logp`), a checkpoint whose forward stashes no flat logits falling back to α / β. Both map onto
  the one event space (HP collapsed; a switch-in by species). On fresh checkpoints `fixed_mass` misses 0 % and `blob`
  4.9–9.0 %: compare the log loss AND the miss rate. `other_label_rate` (F-X5-8) is U4's training metric; the reader's
  per-candidate probabilities are not reported per column yet (OTHER's predicted-vs-label calibration is a follow-up).
- **For the registration (§7.4).** The adoption metric is `per_run.intent_logloss` with `intent_miss_rate` read beside it
  (the miss is a separate column by design, so the registration must say how a miss-rate difference enters the verdict:
  NOT decided here, standing rule 9). Directions are declared (`metrics.PER_RUN_DIRECTION`). Off-pool is n = 0 until a
  ladder bank exists.
- **For U8.** Nothing; the reader needs no GPU.
- **Open (findings).** F-X5-41 (the deflated move prior — FIXED 2026-10-04, role set 15 → 29),
  F-X5-42 (blob's typed-HP label mask, pre-existing), F-X5-43 (Struggle; the bank's unmodeled rows).

**U6 hand-off (read before U8).** Built in U6 (`CHANGELOG.md` "X5 U6"). No model semantics change, no config bump, no
`ARCH_SIGNATURE` bump. F-X5-44's mask is byte-identical in both arms.

- **What exists.**
  - `learner_golden.ARMS` / `arm_args` / `build_arm_learner` / `arm_buffer` / `check_arm` / `arm_coverage` /
    `record_arm`.
  - `python -m agents.training.learner_golden {check,record,rebuild-buffer} --arm fixed_mass`.
  - `learner_golden_fp64.reference` / `violations` / `Fp64Mode`.
  - `parity_probe.perturb_(…, keyed=True)` and `testkit.fresh_model(…, perturb_keyed=True)`.
- **Re-recording rule.** ANY fixed_mass code change that reaches the update moves `arms.fixed_mass`. Re-record it
  with `record --arm fixed_mass --reason …`: it re-measures the fp64 reference, the K9(b) read and the coverage
  counts. A tolerance failure means the fp32 path left its fp64 twin; do not widen the bar. Rebuild the buffer only
  when the observation layout or label schema changes, or after F-X5-45. A blob-side change must leave
  `arms.fixed_mass` alone unless the code is shared.
- **For U8.**
  - `excluded_frac` on the golden is 6.25 % of a 16-row micro-batch; record the real launch's
    `behaviour/excluded_frac`. The ceiling was measured on blob states (`k9_behaviour_exclusion/`), so a fixed_mass
    live share near 0.15 means re-measuring, never raising it.
  - The arm's CPU compute is ≈ 2–3 s per update on 64 rows (blob ≈ 2.5 s).
  - Not covered: compiled and CUDA numerics (K8's regions, startup parity, the canary — U8's first two minutes).
- **Open (findings).** F-X5-45 (rebuild rounding), F-X5-46 (blob entry lacks `init_group_sha256` — FIXED 2026-10-04 by
  the F-X5-41 re-record).

**F-X5-41 fix (2026-10-04, `gen3_smogon_prior_denominator_v1`).** The move prior is `Moves / W`. Both K9 arms were
re-recorded (blob post `c4f63287…`, fixed_mass `355c9da2…`) on REBUILT buffers, because K9(b) failed on the parent-prior
behaviour log-probs. The buffers hold different games (blob 60 / 64 actions, fixed_mass 2 / 64), because the seeded
behaviour learner reads the prior. PROOF (`research_state/measurements/move_prior_golden_2026-10-04/`), three steps:
(A) on the parent's buffers, this tree with the move prior held at the parent's values equals the recorded goldens on
every field; (B) on the rebuilt buffers, the parent tree equals this tree holding the parent's prior; (C) a rebuild
holding the parent's prior replays the parent's buffers (fixed_mass byte-identical; blob's behaviour columns within
F-X5-45's 7.2e-7). Open: F-X5-47 (the UNWEIGHTED species-usage marginal) — FIXED the same day, below.

**F-X5-47 fix (2026-10-04, `gen3_smogon_species_usage_weighted_v1`).** The species-usage marginal is each species'
rating-weighted W. Both K9 arms were re-recorded (blob post `09222426…`, fixed_mass `50659234…`) on REBUILT buffers,
because K9(b) failed on both committed buffers (blob max |Δ log π| 0.0054, fixed_mass 0.0077; both pass with the
parent's marginal held). The rebuilt buffers hold the SAME games (obs, actions, masks and rewards byte-identical); only
the behaviour columns moved (blob 29 / 64 rows, fixed_mass 36 / 64). PROOF
(`research_state/measurements/species_usage_golden_2026-10-04/`), three steps: (A) on the parent's buffers, this tree
with the marginal held at the parent's values equals both recorded goldens on every field (103 / 99); (B) on the rebuilt
buffers, the parent tree equals this tree holding the parent's marginal (103 / 99); (C) a rebuild holding the parent's
marginal replays the parent's buffers byte-identically (both arms). fixed_mass's K9(b) excluded share went 6.25 % → 0 %;
coverage is unchanged (OTHER_species live 36 / dead 28, OTHER_move 60 / 4).

### 8.4 Where each review item landed

| item | resolution | where |
|---|---|---|
| M1 capped presence → infinite BCE | Logistic fixed-size marginals (LML / I-projection), bisection specified and measured, k = 0 / k = n structural, ∂BCE/∂τ = 0 | §2.5, §3.2, §3.9, Decision record |
| M2 max reductions vs I1 / I2 | 41-site census, three classes, I1 / I2 scoped to class E; class M is an owner choice with a recommendation and its second-lever status named | §3.5, §9 M2 |
| M3 OTHER without physics | Measured 57–74 % of hidden mass, recall 0.34–0.52; four options with costs; recommendation; regression stated honestly | §3.3, §3.7, §9 M3 |
| M4 the A − A2 contrast | Reported with what differs; σ ∈ {2.5, 3.43, 4.8} rows on every sizing table, plus 5.21 and 8.0 | §7.1, §7.2, §7.4 |
| M5–M8 A/B statistics | H2H cross primary (pre-registered), δ 3.5, three-look t-scale O'Brien–Fleming with a simulated and re-calibrated constant, non-binding futility, INCONCLUSIVE at every look, panel reported with a 2δ harm flag, purpose metrics across seeds on a group-sequential boundary | §7.3, §7.4 |
| M9 carry every decision | KL stop OFF everywhere (§0, §5, §6, §8.1, §8.3, U5 dropped); threshold and run counts updated; U1 / U6 parts DONE; status line | throughout |
| M10 π's gradient path | Detached into every policy / critic use; justified; what is lost | §3.2 |
| FIX-DURING-BUILD | tie order + rule-8 exclusion (§3.1); OTHER physics contradiction and `switch_branch` undercount (§3.7); OTHER_move and bench E5 bias (§3.1); aux heads on hypothesis seats (§3.4); dex-row cross-check (§3.4); speed-rule definitions and 10M checkpoints (§7.4); build thread count (§7.4); pairing claim (§7.4, F-X5-17); flat-pointer I2 (§3.5); literature (§2); 11–13 agent-days, Opus-tier (§8.3) | as listed |

---

## 9. For the owner: three choices (brainstorm, 2026-10-03 ~17:00)

> **ANSWERED 2026-10-03 (owner): M2 = C, M3 = (c).** Margin (orchestrator, under the owner's "reasonable richness vs budget"): 3.5 pp if P0's σ_h ≤ 2.5, else 4.5 pp. The limitations and the triggers for revisiting each are in [`design_x5_tradeoffs.md`](design_x5_tradeoffs.md).

Each choice gates U3. Recommendations are mine; the costs are MEASURED or ESTIMATED as tagged above.

### M2: what a MAX over uncertain opponents means

Fourteen reductions take a worst case over their candidate moves or mons: "the hardest hit their active can land on
my mon", "does any of them carry Pursuit". With presence-weighted candidates, no max can be both "π = 0 is the same as
absent" and "two half-tokens equal one token". So we choose a meaning.

| option | meaning | pros | cons | second lever? |
|---|---|---|---|---|
| **C. Keep today's presence-scaled max, max_m (π_m · v_m)** (recommended) | "the largest presence-discounted threat" | Exactly today's formula on the move axis (only the weights change, which X5 changes anyway); I1 exact; zero new code paths; the blob is its special case | Ad hoc: not an expectation or a quantile; two 50 % threats of 100 % read as one 50 % threat | **No** |
| A. The expected max under independent presence, E[max] = Σ_(v desc) v_(m) π_(m) Π_{l<m}(1 − π_(l)); for probabilities this is the noisy-OR | "the expected worst hit, given what may be present" | Principled and consistent with §3.2's belief (the I-projection is a product of Bernoullis); I1 exact; continuous across ties | Changes what the op says about the active's REVEALED moves too, in the X5 arm only; a sort + cumprod per site | **Yes**: it changes the move-axis channel for revealed mons, so it would ride on the A/B. Better as its own lever after X26 |
| B. A hard max over hypotheses with π above a threshold | "assume anything likely enough is there" | Simple | A threshold on a continuous π is a rule-8 boundary; counts a 0.31 hypothesis as a whole mon; discards the rest | No, but unsound |
| D. A soft worst case, τ·log Σ π e^{v/τ} (the entropic certainty equivalent; Föllmer & Schied 2002, from memory, UNVERIFIED) | "a risk-averse average" | I1 and I2 both hold (the ToMe identity inside a log-sum-exp) | A new temperature τ to choose: a hyperparameter, so a lever | Yes |

**Recommendation: C for the A/B,** with I2 declared false and pinned by the class-M tests; **A as a named lever after
X26** if the purpose reads show threat mis-pricing. C keeps the A/B to one lever, which is the orchestrator's stated
preference, and no sound option avoids a second lever while changing the meaning.

### M3: what OTHER is to the physics

At a 1:1 budget OTHER holds **57–74 % of the hidden mass** and the list holds only **34–52 % of the true hidden mons**,
under the Smogon prior and under every pool-memorising proxy (§3.3). The live head may be sharper on-pool (UNVERIFIED).

| option | what OTHER (or the list) gets | OTHER's share at r = 1 / 3 / 5 (prior) | cost | pros | cons |
|---|---|---|---|---|---|
| a. A larger budget: + 6 hypothesis tokens (budget (6 − r) + 6) | more concrete hypotheses | 41 % / 31 % / 25 % (recall 0.69 / 0.74 / 0.77); budget 12: 38 / 22 / 12 %; 18: 24 / 12 / 6 % | +6 tokens ≈ +7 % forward FLOPs, op rows ∝ budget, +0.5 agent-day | More concrete physics, better pointer recall | Still leaves 25–41 % unpriced unless combined with (c); a token-budget lever |
| b. Bounded physics: OTHER's op rows = the presence-scaled WORST case over the next 32 tail species | a pessimistic stand-in | unchanged (still 57–74 % of mass) | the op on 32 tail species per row: ≈ 5× the hypotheses' op cost | Never under-states a threat | Max-type (inherits M2); pessimistic by design; expensive |
| **c. Hybrid: OTHER gets today's AVERAGED defender / attacker, computed on the renormalised TAIL distribution only** (recommended) | the blob's own construction, for the tail's share | unchanged, but now PRICED | one `P_tail @ tables` matmul per row: what the blob pays today | X5 becomes a strict refinement of the blob (the blob is X5 with zero hypotheses); every cell prices OTHER on today's semantics (no IMMUNE `e_mult = 0`, §3.7); no new token | The Jensen gap stays on the tail share; it is the "blob again" for that share, deliberately |
| d. No physics (the first version) | nothing | 57–74 % unpriced | 0 | Cleanest separation | **A regression on the physics channel** against today; cells need invented neutral values |

**Recommendation: (c) at the 1:1 budget for the A/B,** so X5 can only add information to the blob's, with one token.
**(a) is the follow-up lever** if U8 / the A/B show OTHER's attention share or mass binding. (b) and (d) are not
recommended. Others considered: OTHER split into a few Smogon-derived type buckets, each averaged (more tokens, a
finer Jensen gap; a later refinement of (c)); sampled tail representatives (nondeterministic, rejected under rule 8).

### The margin against the budget (the simulation says the 3.5 pp design is underpowered)

At the 41 GPU-h ceiling, with σ_h unknown, power at Δ = 0:

| margin | ≈ Elo | σ = 2.5 | σ = 3.43 | σ = 4.8 |
|---|---|---|---|---|
| **3.5 pp (orchestrator's choice)** | 24 | 0.81 | **0.57** | 0.36 |
| 4.5 pp | 31 | 0.94 | 0.76 | 0.51 |
| 5.5 pp | 38 | 0.99 | 0.89 | 0.66 |

Type-I is ≤ 0.047 in every cell, and the expected cost is ≈ 34 GPU-h if X5 is truly equal. P0 (CPU, before any GPU)
measures σ_h and tells us which column we are in. **Recommendation:** keep δ = 3.5 pp if P0 reads σ_h ≤ 2.5; otherwise
choose between 4.5 pp at the same budget and accepting ~0.57 power. A low-power NOT DETECTED keeps the blob, which is
the safe default, but would leave X5 unadopted for reasons of noise.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-03 | Presence weights (proposal) — **SUPERSEDED** by the M1 row below | Iterative capped πps (Tillé 2006): Σπ = k exactly | `min(1, k·q)` (`design_q_head.md` §1): loses mass to OTHER whenever a weight caps | §3.2, §3.9 (first version) |
| 2026-10-03 | Hypothesis content (proposal) | A dex-row table from THE observation encoder, committed beside the code, byte-gated (+ a real-state cross-check, revision) | A torch-side row builder (a second encoder) | §3.4 |
| 2026-10-03 | Presence in attention (proposal) | Log-π key bias in every EXPECTATION-type reduction (ToMe proportional attention), pinned by I1 / I2 (scoped to class E by the revision) | A learned scale (breaks I2); unweighted reductions | §3.5; Bolya et al. 2023 |
| 2026-10-03 | OTHER (proposal) — **physics REOPENED** by the revision (§9 M3) | Entity with a learned + tail-mean embedding; masked structurally | — | §3.3 |
| 2026-10-03 | Token count (proposal) | 61 → 62: OTHER_species new; OTHER_move re-uses the active's E5 seat | +2 seats | §3.1 |
| 2026-10-03 | A/B statistics (draft for the orchestrator) — **SUPERSEDED** by the ORCHESTRATOR rows below | Run-level SE, P0 picks the primary meter, K = 3 → 5 two-look O'Brien–Fleming, matched-wall-clock rule | — | §7 (first version) |
| 2026-10-03 | KL stop (draft for the owner) — **SUPERSEDED** by the OWNER row | A safety guard on the group mean | — | §5 |
| 2026-10-03 | KL stop (OWNER) | **OFF for the X5 A/B and for X26: no stop at all, not even a guard.** X28, the held-out-yield epoch controller, is the named lever after X26. §5's wiring is NOT built; U5 is dropped. | Active stop at 1.5 × target_kl (would fire on 41–74 % of updates, a dose change); a safety guard on the group mean | §5; owner 2026-10-03 |
| 2026-10-03 | X26 by continuation (OWNER) | **YES.** The A/B arms carry the X26 ride-along heads from step 0, and the winning arm's seed-1001 run CONTINUES as the X26 baseline | A fresh X26 launch | §7.4; owner 2026-10-03 |
| 2026-10-03 | A/B budget (OWNER) | **Up to ~41 GPU-h** | Caps at 15 / 25 GPU-h | §7.2; owner 2026-10-03 |
| 2026-10-03 | A/B threshold (ORCHESTRATOR, first version) — **SUPERSEDED** by the review rows below | P0 picks the meter; H2H δ 3.5 at K = 3 → 5, or untaught δ 4.3 at K = 5 → 8; outside panel as a GUARD | — | §7 (first version) |
| 2026-10-03 | Presence bias + pointer (ORCHESTRATOR) | Plain log-π bias (keeps I2 exact on class E); the flat pointer BUNDLED with the tokens (one retrain boundary) | A learned scale; the pointer as a second arm | §3.5 |
| 2026-10-03 | Order (OWNER) | T15, the bottleneck profile and P0 run DURING the X5 build, not after the A/B | Profiling after the A/B (X26 slips about a day) | owner 2026-10-03 |
| 2026-10-03 | F-X5-3 label-writer guard (BUILT, GIGO unit) | Every Rust label writer returns an `Err` (a FAULT) on any lookup it cannot make; species match by DEX NUM | Throw only at `belief.rs:79-81`; keep id matching | §3.8, §8.2; `CHANGELOG.md` "F-X5-3" |
| 2026-10-03 | F-X5-4 K9 harness thread pin (BUILT, GIGO unit) | `build_learner` builds inside `_one_thread()` and restores the caller's count; the banked golden is unchanged | A thread-independent init (changes every init byte) | §6 item 5, §8.2; `CHANGELOG.md` "F-X5-4" |
| 2026-10-03 | F-X5-5 production fresh-build thread pin (BUILT, hygiene unit) | The trainer's fresh build (`construct_fresh_learner`) and the fresh fixtures build inside the ONE shared `single_thread_build`; the caller's count is restored | Pinning only the harness; replacing SB3's orthogonal QR (changes every init byte) | `CHANGELOG.md` "F-X5-5"; `50fdfdc2` |
| 2026-10-03 | **M1 presence construction (ORCHESTRATOR, after review)** | **Logistic fixed-size marginals π = σ(a + τ), τ by a 64-step fixed bisection on a provable bracket; k = 0 and k = n structural; set BCE on the exact logit.** Reason: the capped πps set π = 1 and gave an infinite BCE with a dead gradient on 0.23 % of cold-start decisions (MEASURED); the logistic form keeps Σ = k exactly, never reaches 0 or 1, and is grounded (LML; I-projection). | Capped πps (Tillé); `min(1, k·q)` | §3.2; `x5_revision_2026-10-03/out/m1m3.log` |
| 2026-10-03 | **M10 π's gradient path (ORCHESTRATOR)** | **π DETACHED into the attention biases, OTHER's mass, the flat pointer and every op reduction; δ_θ trained only by the presence BCE.** Reason: RL must not tune π as a gate; the calibration purpose metrics stay a read of the supervised belief. | Undetached π (a separate lever if NOT DETECTED) | §3.2 |
| 2026-10-03 | **M5 primary meter (ORCHESTRATOR)** | **The mirrored head-to-head, pre-registered now; P0 is a planning input only (σ_h, meter variance subtracted); the untaught meter is a reported secondary.** Reason: it is the direct question, "is X5 at least as strong as blob", and pre-registering it removes P0's one-df meter selection (M6). | P0 selects the meter; untaught primary | §7.3, §7.4 |
| 2026-10-03 | **Margin and design (ORCHESTRATOR)** | **δ = 3.5 pp fixed, one-sided α 0.05; group-sequential with estimated variance at 3 / 5 / 8 seeds per arm; O'Brien–Fleming on the t scale (constant ×1.02 after simulation); non-binding futility at Δ̂ ≤ −δ; INCONCLUSIVE at every look; NOT DETECTED at the last.** Reason: δ ≈ 25 Elo is the owner's tolerance; the design is VERIFIED by simulation (type-I ≤ 0.047 at every σ). Its power at σ = 3.43 is 0.57; the margin-against-budget trade goes to the owner (§9). | Known-σ z-tests; a fixed single look; the paired-by-seed statistic (dominated) | §7.4; `gs_sim3.log` |
| 2026-10-03 | **M7 outside panel (ORCHESTRATOR)** | **REPORTED, not gated; a HARM flag to the owner if its point estimate is worse by more than 2δ = 7 pp.** Reason: a second gated NI test would cut the joint power at σ = 3.43 from 0.57 to about 0.25–0.32. | The panel as a guard that must itself be non-inferior | §7.4; `m7.log` |
| 2026-10-03 | **M8 purpose-metric inference (ORCHESTRATOR)** | **Across seeds: one value per run, two-sample t over seeds, tested at the stopping look's group-sequential boundary.** Reason: a battle-clustered CI omits the run term (F-X5-6's error again), Agarwal 2021 does not ground it, and testing a secondary at full α at the crossing look inflates its error (Hung et al. 2007; Tamhane et al. 2010). | A battle-clustered bootstrap as the inference | §7.4 |
| 2026-10-03 | **M2 / M3 (OPEN, owner brainstorm)** | Recommendations: M2 option C (today's presence-scaled max, no second lever), M3 option (c) (OTHER priced by the tail-averaged construction) | §9's tables | §9 |
| 2026-10-03 | **M2 / M3 (OWNER) + margin rule (ORCHESTRATOR)** | **M2 = C** (presence-scaled max); **M3 = (c)** (OTHER priced as the averaged tail); **margin = 3.5 pp if P0 σ_h ≤ 2.5, else 4.5 pp** at ≤ 41 GPU-h, fixed before P0 reports. U3 is unblocked. | M2 A / B / D; M3 (a) / (b) / (d); 3.5 pp at power ≈ 0.57 | §9; [`design_x5_tradeoffs.md`](design_x5_tradeoffs.md) |
| 2026-10-03 | **Margin FIXED by the pre-committed rule (ORCHESTRATOR, after P0)** | **δ = 3.5 pp.** P0 (`df27f701`, `measurements/x5_p0_h2h_2026-10-03/`) measured σ_h = 0.33 (post-boundary {A2, A′, B}, 2 df, CI [0, 2.56]), 0.53 (the pure seed pair), 1.63 (all four, 3 df) and 2.12 (the conservative {A, A2, A′}); every point estimate is ≤ 2.5, so the rule gives 3.5 pp. Power at 3.5 pp: 0.905 at σ_h 2.12, about 1.0 at ≤ 0.53. The registered design already estimates σ in-experiment with t-boundaries, so a wide prior CI costs power, never the type-I error. P0 also found that the untaught meter's run-to-run differences do NOT transfer to the head-to-head (slope 0.26; A2 − A′ has opposite signs on the two scales), which supports the head-to-head as primary. OPEN for the build: a multi-cell h2h engine (one engine, weights loaded per cell) so the A/B's 9 / 25 / 64 cells don't each pay a 114–167 s engine start (F-P0-4); folded into U0 | 4.5 pp | P0 result.md; §7.4 |
| 2026-10-03 | **U2 T0 hypothesis builder (BUILT)** | **`agents/model/hypothesis_set.py` under `--belief-tokens fixed_mass` (config v136, STRUCTURAL, default `blob`, production `blob`; no ARCH_SIGNATURE bump).** V = the dex table's valid nums minus the revealed (F-X5-21); τ by 64 fixed bisection steps on the provable bracket under `no_grad` (MEASURED max \|Σπ − k\| 2.7e-15 fp64 / 1.4e-6 fp32); k = 0 / k = n structural; one STABLE argsort of −π, ties to the lower num; hypothesis j → the j-th hidden slot; OTHER = the tail's mass, logsumexp log-mass and a rest + map(tail-mean) embedding; the active's move group at k_m = 4 − r_m over its legal moves, revealed pinned first. Supervision: the `hidden_team_set` row (presence BCE + BeliefHead's set BCE on its hidden-slot-mean logits + hypothesis-seat moves, supervised iff present), exclusive with `hidden_team`, same coefficient. Built from a private seed out of `IsolatedLinear`s, so non-X5 init bytes equal blob's. `requires` gains `opp_belief_slots`. In U2 the set is STASHED only (U3 wires it in). | `torch.topk` (unspecified tie order); a separate presence-BCE coefficient (a second knob, no evidence for one); Hungarian moves on hypothesis seats (the matching is retired); an `nn.Linear` δ_θ (SB3 re-draws it from the global RNG, moving every later init) | §3.2, §3.8, §8.3 U2 hand-off; `CHANGELOG.md` "X5 U2" |
| 2026-10-03 | **U1 dex-row table (BUILT)** | **One SYNTHETIC input to the encoder's own slot writer** (`encoder::hypothesis::hypothesis_slot` → `slot::populated_slot`, the writer the real team loop uses): `PMon::from_species` + `100/100`, its `mon_view`, an empty `SideTrackers`. One row per BASE-FORM species at its num, committed as JSON beside the model code with a sha256 the loader re-derives; a `sim` byte gate; the real-state cross-check over the 386 species (constructed teams) and the bridge corpus. **The declared exclusion list is the mon's on-field state** (status, HP fraction, status counters, sleep belief, recency, last action, active) **plus an item / ability / move block the field revealed, by the real row's own flag** — wider than the first version's three entries, which missed blocks that differ legitimately at a real first appearance. | A torch-side row builder (a second encoder); a full synthetic battle state through `BattleVersion::encode` (far more synthetic surface for one slot); a tolerance on the cross-check (rule 8) | §3.4; `designs/rust_sim/encoder.md` §10; `CHANGELOG.md` "X5 U1" |
| 2026-10-04 | **U3 open items (ORCHESTRATOR)** | **F-X5-23:** OTHER's embedding is the π-weighted mean over the WHOLE tail (no second selection boundary; consistent with M3 (c)). **Tier-0 F4 (a):** OTHER's averaged attacker is the renormalised tail's EXPECTED attacker stats, the `P_tail @ tables` construction the blob uses for the defender. **F4 (b):** at a max-type site OTHER enters with presence P(at least one tail mon present) = 1 − Π(1 − π_i), exact under the I-projection's independent Bernoullis, in [0, 1], continuous (Tier 0 capped the mass at 1). **F-X5-24:** MoveBelief keeps its per-move BCE; the move-side set BCE is a follow-up. **F-X5-22:** unchanged in U3 (U8 measures the GPU compile). **Versioning:** no `ARCH_SIGNATURE` bump in U3; the adoption break decouples the version floor from the signature (D-L1, provisional, PENDING). | A tail-32 mean (a second boundary); the mass capped at 1 at a max site; a move-side set BCE now | §3.3, §3.8, §8.2; `design_x5_tradeoffs.md` §4 |
| 2026-10-04 | **U3 part 3 open items (ORCHESTRATOR)** | **F-X5-26:** under fixed_mass the opponent ACTIVE's MoveBelief reinjection uses the DETACHED π_m, not sigmoid weights (M10: RL cannot tune the belief as a gate there; the move head keeps its BCE; the lost RL gradient through that row is the stated cost). **F-X5-27:** `BeliefSlots` is not built under fixed_mass (a parameter that never gets a gradient); blob keeps it. **F-X5-28:** closed by part 3 (OTHER's edges + op column). **No `ARCH_SIGNATURE` bump** (D-L1 pending, decoupled). | Sigmoid reinjection at the active; building an unused `BeliefSlots` | §3.4, §8.2, U3 part-3 hand-off |
| 2026-10-04 | **U3 part 3 (BUILT)** | The op runs on the HYPOTHESIS context with an `OpRoster`: hypothesis DEFENDERS as the per-slot one-hot expected-latent read, **P(KO) UN-nulled** (a pristine concrete species; only OTHER's averaged defender keeps the null); every live mon an ATTACKER on its own fixed-mass move presence (k = 4 − r) in one stable per-mon order; "alive" = `opp_addressable` in every opponent-slot gate; `p_pur_vs_us` presence-scaled over mons with OTHER at `other_any`; Beat Up by π / k; bench E5 seats presence-aware; the per-mon cuts in `near_tie_rows`. OTHER = an OTHER-mode pass of the same kernels on the tail's `P_tail @ tables` (defender, E[base], E[STAB], E[speed], E10 moves at k = 4), written to its seat's edges for D1 / C1 / C3 / D4 / V. Blob byte-identical. | P(KO) nulled for hypotheses (discards the species-exact KO); π-scaling per-(seat, mon) cells (double-counts the key bias); a separate OTHER physics kernel (a second copy of the physics — drift risk); OTHER's mass at a max site; OTHER edges via zero cells (the map would turn them into its learned bias) | §3.4, §3.5, §8.2 F-X5-29..33, U3 part-3 hand-off; `CHANGELOG.md` "X5 U3 part 3" |
| 2026-10-04 | **U3 parts 1–2 (BUILT)** | Hypothesis tokens = THE `PokemonEncoder` on the hypothesis context (dex rows in the hidden slots, every mask REAL) + `hypothesis_marker`; OTHER_species one extra trunk seat after the entity seats; log π (detached) on every opponent key in the trunk and the four class-E pools (FLOAT masks); MoveBelief's unrevealed population supervised iff the hypothesis species is present; the active's move axis from ONE order with the fixed-mass presence as the op's class-M weights; a revealed Hidden Power's seat priced as its typed mixture through an extended seat axis; K < 4, a K mismatch and candidate truncation refused under fixed_mass. Blob byte-identical. | Re-encoding through a torch-side row builder (a second encoder); OTHER inserted between the team block and the global token (shifts every edge offset); 237 priced as a move (BP 0, typeless); the dominant typed channel as the HP seat (a selection on a continuous weight) | §3.4, §3.5, U3 hand-off; `CHANGELOG.md` "X5 U3 part 1 / part 2" |
| 2026-10-04 | **U7 readers (BUILT)** | The intent read on a COMMON EVENT SPACE (a move by num with every Hidden Power ONE event; a switch-in by species); each arm's P(event) through its own heads' CONTENT (blob's hidden slot = BeliefHead's species posterior over V, the target β is trained on; fixed_mass reads U4's flat pointer, a slot's content its hypothesis, OTHER's the renormalised tail); a miss its own column, the log loss over covered rows only; Struggle excluded (forced). Blob presence = BeliefHead's hidden-slot mean through the same fixed-size construction (§4.2 R3's rule, applied to every presence read); a DERIVED OTHER for blob and the prior; P̂ = MoveBelief's sigmoid inclusion posterior; the role set derived at read time and stamped. | A typed-HP event (the type is not a choice; F-X5-42 shows the label mismatch it would inherit); flooring a miss (hides coverage); blob's hidden switches as an automatic miss (discards what its belief knows); the per-mon fixed-mass π_m as P̂ (unsupervised; not the blob's quantity); a frozen role list (would hide F-X5-44's fix) | §4.3, §7.4 as built; U7 hand-off; `CHANGELOG.md` "X5 U7" |
| 2026-10-04 | **F-U6-1 closed: the cross's engine (h2h agent)** | the X5 cross plays on ONE engine of two T2 slot groups (one per arm), each declared with only the slot its side needs, and one eval core per (player arm, opponent arm); the h2h game protocol unchanged | two full groups (both roles each: twice the slots and lanes for no cell); an executor change to route one core to either group (shared with training eval) | `measurements/h2h_cross_2026-10-04/` |
| 2026-10-04 | **U4 the flat opponent pointer (BUILT)** | One list (K seats · OTHER_move · six switch targets · OTHER_species), one shared scorer + the DETACHED log π, one softmax; α / β retired in the fixed_mass arm by the policy's `_build` AFTER SB3's ortho draws (no non-X5 init byte moves, no state_dict key, no optimizer slot); labels from the existing intent label (OTHER for members only, F-X5-34; a typed HP label names a revealed HP's seat); the cells re-expressed exactly (guarded logsumexp; OTHER_move priced by the tail contraction on the full move axis; OTHER_species by the OTHER-mode D1 pass and the tail's P(Ghost)); `seat_live` in all four consumers under fixed_mass (F-X5-35); B re-based (`FlatOppEffectEnsemble`); `opp_intent/other_label_rate`; `fixed_mass` requires `entity_tail_seats`. Blob byte-identical. | Not building α / β at all (their missing ortho draws would shift every later initial byte); keeping them built (parameters with no gradient, F-X5-27's rule); OTHER_move priced as a zero row or by a top-N tail (a selection boundary); labelling every out-of-seat move OTHER (a label OTHER's mass cannot carry); `seat_live` in both arms (moves production) | §3.5, §3.6, §3.7 "As built (U4)", §8.2 F-X5-34..40, U4 hand-off; `CHANGELOG.md` "X5 U4" |
| 2026-10-04 | **U6 the K9 fixed_mass golden (BUILT)** | A second entry `arms.fixed_mass` beside the untouched blob entry: production + the lever, NAME-KEYED perturbation, its OWN Rust-collector buffer (the blob buffer's behaviour log-probs are blob's — K9(b) 0.51; run seed 18 for §6.4's coverage); `init_group_sha256`; an fp64 reference of one R1 micro-step (same code under `Fp64Mode`, rule-8 rows excluded, declared bars ≥ 10× measured); teeth = four X5 plants fail it and leave blob, one blob plant fails blob and leaves it; δ_θ / B isolation through the real update; F-X5-44 (the dead OTHER's uniform order counted as ties) fixed byte-identically | Reusing the blob buffer (K9(b) cannot pass: stale log-probs); order-keyed noise (no shared group comparable with blob); fp64 `train()` end to end (Adam's sign-like first step amplifies fp32 noise on near-zero gradients — not a fixed bar); widening the excluded-share ceiling instead of fixing the dead-OTHER order | §6 "As built (U6)", §8.2 F-X5-44..46, U6 hand-off; `CHANGELOG.md` "X5 U6"; `designs/training/learner_gates.md` |
| 2026-10-04 | **A/B Amendment 1: blob arm first (OWNER-approved; registered before any launch)** | **The blob arm's look-1 seeds 1001–1003 run first at commit P_blob; the fixed_mass seeds follow at P_x5 after the cost fixes.** Preconditions: blob-path identity across the pins (the K9 `arms.blob` entry, the `data/pokemon/` hashes, the obs golden, `OP_SEMANTICS`, reward constants), else look 1 is INCONCLUSIVE; s from a paired ABAB benchmark at P_x5; U8 gates fixed_mass only; snapshot-ladder updater OFF in all arms (F-ED-18); h2h protocol frozen; no CPU eval lane. Reason: fixed_mass fails U8 and four launch refusals (`d7120efa`), and the blob arm is production. | Waiting for the fixed_mass fixes with the GPU idle; interleaving at one commit (impossible until the fixes land); s from the arms' own walls (confounded by drift, F-XC-6) | §7.5; owner 2026-10-04 |
| 2026-10-04 | **A/B Amendment 2: two ORACLE REFERENCE arms (OWNER)** | **Oracle-species and oracle-full (backlog X32), the facts written into the observation's opponent block, so they enter the shared trunk; 3 seeds each at P_x5; played in the same mirrored cross; REPORTED as the ceiling C and the headroom captured H = Δ / C (only when C clears the floor); never gated, §7.4's rule unchanged.** Reason: non-inferiority says whether the tokens cost strength, not how much belief could buy; the oracle bounds that at this recipe. | A separate experiment (owner: an arm of this one, it is the ceiling); a gated superiority test vs the oracle (spends α, changes the registered design); side inputs to the policy or value head (owner: the shared trunk) | §7.6; owner 2026-10-04 |
| 2026-10-04 | **A/B Amendment 3 (OWNER + orchestrator)** | **(a) oracle cells in TWO modes: one-sided clairvoyance (primary for C) and both-sided reveal (secondary, descriptive; the non-oracle seed is out of distribution). (b) Purpose metric (1) = the CONDITIONAL log loss over blob's named set E_row with both arms renormalised over it; coverage (fixed_mass outside-mass calibration vs blob's miss rate) reported; the as-built metric descriptive. (c) Observability is a design criterion: at similar cost, prefer what makes the model's state readable.** Reason: the as-built metric rewards blob's forced over-certainty and drops its misses (owner's finding). | Each arm on its own covered rows (biased); flooring blob's misses (the floor decides the answer); both-sided reveal as the primary oracle read (out of distribution for the non-oracle side) | §7.7; owner 2026-10-04 |
| 2026-10-04 | **Oracle reveal BUILT, `species` level (`--oracle-reveal`)** | **An encoder-level reveal: the opponent block's unseen tail is the encoder's own never-seen-mon row (`hypothesis_slot`) of each true unseen species in dex-num order; the seen prefix and every other cell are `off`'s; matched on reveal by dex num. Resume-immutable flag (config v137); symmetric by construction; scripted bots unaffected; the fork arm refused; offline play tools refuse an oracle checkpoint. PIN CLARIFICATION: the oracle seeds run at their OWN commit P_oracle (the first main commit carrying this build), not P_x5, under the SAME blob-path identity precondition against P_blob = e5e660dd (§7.5), which this build satisfies because the reveal is inert when off: the off obs / mask / label bytes are pinned to a digest recorded before the build, and the K9 learner golden (both entries), the obs golden, `data/pokemon/` and the reward and critic constants are unchanged.** Reason: the reveal enters the shared trunk through the observation (owner), and an encoder-only change touches exactly the declared cells. | Pre-populating the reading's opponent roster or synthetic `\|poke\|` lines (puts unseen mons into the view, the trackers and every label, each a "seen" leak); a Python-side row rewrite (a second slot layout); a `structural` flag class (would put an observation mode on the ARCH surface and in `check_compatible`) | §7.6 as built; owner 2026-10-04 |
| 2026-10-04 | **Oracle reveal BUILT, `full` level** | **`--oracle-reveal full`: the opponent's whole set (four moves, item, ability, nature / EVs / IVs with `spread_known` 1, and `hp_revealed` 1) written into the observation's opponent block as already-known facts, through the same encoder-level mechanism: an unseen mon's slot is the row of an OWN mon of that set, a seen mon keeps everything play has revealed and gains only the facts play has not (`Oracle::overlay`). Verified against the OPPOSING chain's own-team slot of the same mon (the independent oracle), with the throwing `FULL_SLOT_CELLS` guard; `off` and `species` byte-identical (pinned). PIN CLARIFICATION as for `species`: P_oracle is the first main commit carrying `species`; the `full` seeds run at the commit that carries this level (the first main commit with `full`), each under the SAME blob-path identity precondition against P_blob = e5e660dd, which holds because `off` is byte-identical.** Reason: the full set is the ceiling's second reference (§7.6), written exactly like an own mon's known facts so the trunk reads it in the representation it already knows. | Pre-writing the set into the reading (puts hidden facts in the view and trackers); overwriting a seen mon's observed moves / item with the preview (a stale value fighting a live one); `spread_known` through a new flag (the view already carries it) | §7.6 as built (full); owner 2026-10-04 |
| 2026-10-04 | **A/B Amendment 4: blob seeds 1004–1005 run early (OWNER)** | **Look 2's two added blob seeds run now at P_blob, after the oracle seeds; not read at look 1.** Reason: the GPU would idle until the fixed_mass fixes land; look 1 almost surely says CONTINUE; a pre-declared seed list run early is not optional stopping. | Alternating look 2's seeds (GPU idle about a day) | §7.8; owner 2026-10-04 |
| 2026-10-04 | **Oracle seeds run at a commit carrying `gen3_r1_unmoved_param_v1` (oracle canary FATAL fix)** | **Every oracle seed (species and full) runs at a commit that carries the R1 unmoved-parameter rule. At 401873b5 the canary FATALs at update 10 by construction: the oracle's all-PAD belief labels leave the zero-init species head at exactly 0.0, and the trained bar reads its fresh-weights fp32 noise (1.40e-2 vs 9.88e-3) as a miscompile. A pinned resume of such a seed FATALs at its startup gate, so the killed `rb_x5ab_oracle_sp_s1001` relaunches FRESH. The same blob-path identity precondition against P_blob = e5e660dd holds: the fix changes only R1's verification (the gate and the canary), with no training arithmetic, no K9 golden entry, no observation byte and no data file touched.** Reason: the FATAL was noise on a parameter training never moved, measured on the CPU (eager fp32 errs 1.51e-2 against float64 there; the compiled arm errs 4.4e-7), and the rule still holds that head to the trained bar on its own perturbed rung. | Running the oracle seeds with `--compile-trainer` off (a different learner from the blob arm's); excluding the head from the canary (a miscompile on its path would pass); a per-run bar override | ledger 2026-10-04 "oracle-species canary FATAL"; `measurements/oracle_canary_2026-10-04/`; `designs/training/compile_flags.md` |
| 2026-10-04 | **Oracle-full seeds run at a commit carrying `gen3_r1_unmoved_init_v1` (the unmoved rule reads the init record)** | **Every oracle-full seed launches FRESH at `77245f51`, the commit that lands `gen3_r1_unmoved_init_v1`, or a later one with the same blob-path identity against P_blob = e5e660dd. That rule judges a parameter BIT-IDENTICAL to its fresh-build value at the FRESH bar, with its own perturbed rung at the TRAINED bar. The record is written by the trainer's fresh construction and rides in every checkpoint, so only a FRESH launch at that commit has one. A resume from an older checkpoint keeps the exactly-0.0 rule.** Reason: in the species FATAL checkpoint 7 oracle parameters never moved, and the zero rule saw only 4 of them. The ortho-init `moves_head.weight`, the belief LayerNorm weight and `unknown_slot_emb` were judged at the trained bar on fresh-conditioned gradients (CPU eager vs float64: 9.4e-4 / 6.4e-4, over 1,000x a trained parameter's). `full`'s fresh build and its no-op losses are `species`'s, so its unmoved set is predicted to be the same (UNVERIFIED until a `full` checkpoint exists). The rule changes R1's verification only: no training arithmetic, no RNG draw, no K9 golden entry, no observation byte. The species relaunch at 8b8fbac0 trains identically. | Launching oracle-full at 8b8fbac0 (the zero rule covers the one parameter that FATAL'd, but the ortho-init dead head stays at the trained bar, inside a ~10x CPU-eager margin); deriving the init on a resume by rebuilding at the recorded seed (trusts the rebuild to equal the run's own init, which a code change between commits breaks silently) | ledger 2026-10-04 "R1's UNMOVED rule now reads the run's INIT RECORD"; `measurements/oracle_canary_2026-10-04/`; `designs/training/compile_flags.md` |
| 2026-10-05 | **K9(b) tie identity: oracle-full returns to `--behaviour-check fatal` at the commit landing `gen3_behaviour_tie_identity_v1`** | **YES for any launch at that commit or later.** The oracle-full stop (excluded 0.220 at the first probe, max \|Δ log π\| exactly 0) was a first-update artifact. The zero-init action scorers make every logit 0, so none of the 514 excluded rows (25.1 % on CPU) could move log π. The rule now judges every row while the scorers are zero (SELECTION-FREE). An argmax tie between candidates whose gathered payload is bit-identical is no tie (PAYLOAD IDENTITY). Oracle-full's live updates 1–11 read 4.98–8.11 % under the old rule, and the CPU perturbed read is 3.6 % under the new one, so the 0.15 ceiling stays and no arm gets its own. The running seeds are pinned at `77245f51` in warn mode with the strict mismatch stop; nothing in their reads changes. | An arm-specific ceiling for oracle-full (it would widen the ceiling to make a first-update coverage failure pass, and the share after update 0 never needed it); a run-time flip test of every tie (complete only over every subset of a row's units, and an extra forward per variant) | `designs/training/learner_gates.md` (b); `measurements/k9_tie_identity_2026-10-05/`; ledger 2026-10-05 FINDING + FIX |
| 2026-10-06 | **K9(b) consumption: an X5 sort pair counts only where its caller reads the ORDER (`gen3_behaviour_tie_consumed_v1`)** | **YES for any launch at that commit or later; nothing about a running seed changes.** `rb_x5ab_fm_s1006` stopped at update 1 (excluded 0.281, max \|Δ log π\| 2.4e-7) on ties inside the per-mon move orders' top six, which the op reads as a set (bit-identical under any permutation on 2,048 real rows; the cut pair moves 2,002 rows). Each caller of `stable_order` now declares its reading (SetCuts at the op's cuts; in order up to k for the species; the move group unchanged). CPU at the dumped weights: 15.2 % → 5.9 %, 0 distinct rows cleared at five states; the update is bit-identical (fixed_mass golden `changed: []`). The ceiling stays 0.15 and a judged mismatch stays FATAL in every phase | A declared EARLY PHASE whose ceiling is reported, not fatal (it trades away the ceiling's protection exactly while ties are commonest, and blob_s1005 read 0.148 at its fourth probe under the rule before, so "early" is not where the margin is thin); a bound on \|Δ log π\| from scorer weight norms (at update 1 the bound is ~0.2, three orders over the 1e-4 bar) | `designs/training/learner_gates.md` (b) CONSUMPTION; `measurements/k9_early_probe_2026-10-06/`; ledger 2026-10-06 FINDING + FIX |
| 2026-10-05 | **F-XC-4 root-caused; the compile gate names a non-finite gradient and proves its arms independent (`gen3_gate_nonfinite_named_v1`, `gen3_gate_independent_arms_v1`); a fixed_mass `--compile-trainer` launch stays REFUSED** | **The gate fix lands now (no arithmetic moves in any arm). Making fixed_mass compile is a separate, arithmetic-touching unit; it gates every fixed_mass seed, beside F-XC-2 / F-XC-3, the cost fixes and the new logger-key crash.** The 'cosine 0.000000' was a NaN compiled backward on CUDA (41 parameters, 8 of 8 compiles at HEAD) that `_cos` read as orthogonal; Inductor's `remove_noop_ops` is where it enters (fusion-dependent); R1 IS one compiled graph under fixed_mass, so X5's +16.7 % is a compiled cost. F-XC-5 on the GPU at HEAD: 0.090–0.113 < 0.15 (eager learner; see §3.6). | Run fixed_mass with `--no-compile-trainer` (an uncompiled learner is a ~1.75–2× slower arm, not the design's cost); disable `remove_noop_ops` globally (moves every arm's compiled kernels; needs every gate and the benchmark — the candidate fix unit) | `measurements/x5_fxc4_compile_gate_2026-10-05/`; `compile_regions_independence_test.py` |
| 2026-10-05 | **§7.7(a) BUILT: the per-side oracle reveal in `main.h2h`** | **One mode per plan (`off` / `one_sided` / `both_sided`); each cell's (p1, p2) levels DERIVED from the mode and the two checkpoints' recorded levels; the Rust spec's `oracle_reveal` takes a per-side pair (symmetric still a string); one eval core per (groups, levels); each mode its own eval protocol + `compute.oracle_reveal` on every row; oracle-vs-oracle refused. `off` byte-identical (recorded at `c7b4d03e`).** Reason: §7.7(a) needs the clairvoyance read through the training core's own encoder path, and pooling an oracle row with an `off` row (or with the other mode) must be impossible. | Typed per-side levels on the CLI (a wrong level would be a typo away); one protocol with the levels only in `compute` (readers would pool the modes); a Python-side observation patch (a second encoder) | §7.7 "As built — the per-side reveal" |
| 2026-10-05 | **F-XC-4 FIXED for fixed_mass by an index-selected max (`gen3_fm_index_max_v1`); blob untouched** | **Under fixed_mass the incoming direction's ten channel maxima over the full candidate sweep are taken by `damage_op.max_by_index` (the value at the detached argmax), not `amax`. The value is bit-identical; the backward is a scatter at a saved index, so no recomputed float has to equal a saved max. Scope: the `fixed_moves is not None` branch of the incoming direction only. The blob arm's R1 and T2 compiled code is byte-identical before and after (dynamo graph + every Inductor module hashed with the caches off), and its K9 golden passes, so the §7.5 identity precondition against P_blob = e5e660dd holds. Eager change for fixed_mass: on an exact tie the gradient goes whole to the first maximal element (amax split it); the fixed_mass K9 golden is unchanged. CUDA: compiled gradients finite and the real startup gate PASSES in 4 of 4 fresh processes; planted NaN / gradient miscompiles still FATAL.** Reason: the defect is `amax`'s tie-count backward over a tensor Inductor recomputes and Triton's FMA contraction rounds differently (finite with contraction off); selecting by index removes the float equality the backward depended on. | `remove_noop_ops` off for R1 (a global Inductor pass change for the arm, and it only moved which nodes were recomputed — the class stays); Triton FMA contraction off for fixed_mass (process-global, slower kernels, and it removes one rounding source rather than the equality); a newer torch (no release with a fix identified); respelling every `amax` in the model (blob's would change, voiding §7.5) | `measurements/x5_fxc4_nanfix_2026-10-05/`; `designs/training/compile_flags.md` (F-XC-4); `damage_op_index_max_test.py`, `compile_regions_fixed_mass_cuda_test.py`; ledger 2026-10-05 FINDING + FIX |
| 2026-10-05 | **A/B Amendment 5 (OWNER)** | **fixed_mass launches at s ≤ 50 % (refusals still FIXED, never waived); strength read at BOTH matched steps and matched wall-time (15M/(1+s)), each at the registered boundaries; adoption is the owner's decision with both reads, purpose metric (1), s and GPU-hours; "either read suffices" is disallowed.** Reason: the owner must decide what to do given the cost, not just the strength. | The +5 % budget as a launch gate (the cost work may not reach it); the 5–15 % / > 15 % speed rule; automatic adoption | §7.9; owner 2026-10-05 |
| 2026-10-05 | **The hypothesis encoding is the per-row pass's exact split, gathered (`gen3_x5_hyp_gather_v1`); fixed_mass only, blob untouched** | **The species-only columns of `PokemonEncoder`'s two first Linears are computed once per forward over the 400-row dex table (over the slots' own dex rows when B·6 < 400) and gathered by hypothesis species; the row-level columns (clock, weather, fainted, hazards, screens, the active-context scatter) once per row; the rest of the encoder per OPPONENT slot only. Equal to the per-row pass up to fp32 reassociation (≤ 2.9e-6 on real rows; ≤ 6.2e-15 at fp64); the fixed_mass K9 golden re-recorded; blob's R1 and T2 compiled code byte-identical.** Reason: the encoding pass was X5's largest single cost; the brief's premise (a hypothesis token depends on its species alone) is false — five row-level features enter — so only the first layers split exactly. Effect: the encoding 10.09 → 4.38 ms per micro-batch; `train_ms` +19.5 % → +12.8 %; T2 +46.1 % → +45.6 %; headroom 1,106 → 1,192 MiB. §3.6's `train_ms` and T2 lines still FAIL. | Encoding the species once with the row features dropped or frozen (a cheaper token, but it changes what is computed: a design change for the owner); a one-hot matmul gather (deterministic backward, ≈ 0.5 ms; the atomic backward did not show in the profile); compacting to hidden slots only (≈ 12 % of opponent slots; needs dynamic shapes or a capped buffer) | `measurements/x5_hyp_gather_2026-10-05/`; `hypothesis_encode.py`, `hypothesis_encode_test.py`; ledger 2026-10-05 MEASUREMENT + BUILT |
| 2026-10-05 | **F-XC-3 FIXED: the parity gate's eager reference leaves no per-forward state on a served replica (`gen3_reference_state_released_v1`); s MEASURED +16.7 %; fixed_mass LAUNCHABLE under Amendment 5** | **`policy_reference` runs under `decision.forward_state_released`, which empties every module attribute the forward replaced (a stash to a fresh instance, a tensor or tensor tuple to `None`, a created one removed). The paired ABAB benchmark gives s = +16.7 % (block pairs 16.4–17.1 %), so the matched wall-time checkpoint is 12M.** | Raising the 1 MiB slot-load tolerance (an override; Amendment 5 forbids waiving a refusal); keeping the stash but pre-sizing it at startup (still row-sized state per replica for the run, and it re-breaks at a new bucket); clearing only `ExtractorStashes` (the op stash, `last_move_tokens` and `EntitySeats.last_cand` also outlive the forward) | `measurements/x5_launchable_2026-10-05/` (mechanism, blob + fixed_mass compiled-code identity, the real launch), `measurements/x5_paired_speed_2026-10-05/` (s); `reference_state_test` fails on revert (both arms) |
| 2026-10-06 | **LOOK 1 READ (registered rule, no decision by anyone yet)** | **Matched steps: CONTINUE** (Δ̂ −1.31 pp [−6.09, +3.46], t 1.27 < 5.761 on 4 df) → look 2 adds seeds 1004–1005 per arm (blob's are banked; fixed_mass's must be trained). **Matched wall-time (fixed_mass 12M vs blob 15M): FUTILITY STOP, verdict NOT DETECTED, INFERIOR** (Δ̂ −6.78 pp [−9.28, −4.28]; non-binding; Amendment 5: a cost-per-GPU-hour deficit at s = +16.7 %, not evidence about the representation). Purpose metric (1) not tested. **Oracle ceiling** (one-sided): C_species +2.44 [+0.31, +4.57], C_full +2.18 [+0.53, +3.84], C_full − C_species −0.26 [−1.92, +1.40] pp; both below the replicate floor F = 4.57 pp, so H is not reported ("the ceiling is within the noise: belief representation has little strength leverage at this budget"). The estimator (`main.h2h.cross`, `bcb0296c`) was committed before any look-1 cell entered the ledger; it fixes the choices the rule leaves open (two-sided intervals, C's lower bound = the two-sided lower end, F = max(P0's 4.57, the blob columns' spread), rule 8 at every boundary). Adoption stays the owner's (§7.9) | — (a read, not a choice) | ledger 2026-10-06 READ; `measurements/x5ab_look1_2026-10-06/` |
| 2026-10-06 | **Amendment 3(b) reader (BUILT): purpose metric (1) = the CONDITIONAL log loss over BLOB's named set; a fixed_mass run is scored on its PAIRED blob run's set** | E_row = the blob checkpoint's structural support on the common event space (its namable E4 seats + the switch-ins its β-legal slots' contents support). The design text defines E_row as "blob's named candidate set", and a fixed_mass checkpoint has no blob heads, so **E_row on a fixed_mass row is a REFERENCE blob run's, read on the same bank row: fixed_mass seed s ↔ blob seed s** (the seed id is used only to assign supports, never to pair the t). `infer` enforces a one-to-one map onto the control group, so both arms are scored over the same collection of supports. Rule 8 excludes and counts the reference's seat-cut near-ties, the arm's own read ties, and denominators ≤ 1e-12. A zero-mass in-set event makes the value None. Coverage and the as-built form are reported, descriptive. **FINDING for the owner:** the text does not say WHICH blob run's set a fixed_mass run is scored on, so seed-id pairing is this build's choice | (i) a model-free "blob-like" set, revealed moves + the top-6 by the shared Smogon prior: blob's seats are the top-K of its LEARNED move belief (`damage_op._opp_candidate_weights`), so a prior-based set is not blob's set; (ii) one reference blob run for every fixed_mass run: one run's set idiosyncrasy would shift the whole treat arm; (iii) each fixed_mass run averaged over every blob run's set: also faithful, but the read depends on all n blob runs and changes as seeds are added | §7.7(b) "As built"; ledger 2026-10-06 BUILT. **SUPERSEDED the same day by the next row** (fixed_mass scored on every blob run of the look, the mean) |
| 2026-10-06 | **Metric (1) for a fixed_mass run = the MEAN over EVERY blob run of the look (ORCHESTRATOR, under the owner's delegation; supersedes the seed-id pairing above)** | A fixed_mass run is scored on each blob run's named set of the same look, and its metric (1) value is the MEAN of the conditional log loss over those sets. Blob runs keep their OWN set. `infer` requires each fixed_mass read to reference exactly the look's blob runs, the same set for every fixed_mass run, all of them. **Reason:** §7.4 says "seeds do not pair runs" (two GPU runs of one seed diverge from the first update), so fm s ↔ blob s is arbitrary; averaging over all of the look's blob sets mirrors the cross design (every fixed_mass seed against every blob seed) and leaves no arbitrary choice. **The one-seed smoke values (s1001 only, previous row's build) were seen BEFORE this choice; the choice follows from §7.4's principle, not from them.** The registered rule and boundary are UNCHANGED: one value per run, the across-seed two-sample t on 2(n − 1) df, tested only after a NON-INFERIOR matched-steps verdict, at that look's t-boundary (5.761 / 2.683 / 1.874) | seed-id pairing (arbitrary under §7.4); one reference blob run for all (one run's set idiosyncrasy shifts the whole arm) | §7.7(b) "As built"; ledger 2026-10-06 BUILT (mean over blob sets) |
| 2026-10-06 | **LOOK 2 READ (registered rule, no decision by anyone yet)** | **Matched steps: CONTINUE** (Δ̂ −0.97 pp [−3.36, +1.41], t 2.441 < 2.683 on 8 df; n = 5 per arm, the full 5 × 5 cross: look 1's 9 cells reused, 16 new) → look 3 adds seeds 1006–1008 per arm (39 new cells, boundary 1.874 on 14 df). **Matched wall-time (fixed_mass 12M vs blob 15M): FUTILITY STOP, verdict NOT DETECTED, INFERIOR** (Δ̂ −6.89 pp [−8.88, −4.90], upper one-sided 95 % −5.29; non-binding; Amendment 5: a cost-per-GPU-hour deficit at s = +16.7 %, not evidence about the representation). Purpose metric (1) not due (it follows a NON-INFERIOR matched-steps verdict only). Played at `131f3412` after a reviewed engine drift (the request spec byte-identical to look 1's, no game-path change). Adoption stays the owner's (§7.9) | — (a read, not a choice) | ledger 2026-10-06 READ (look 2); `measurements/x5ab_look2_2026-10-06/` |
| 2026-10-06 | **ADOPTION RULE pre-committed by the OWNER, BEFORE look 3 is read** | **"Hypothesis tokens are a go as long as they aren't worse. I am willing to take a risk on it."** Operationalised: X5 (fixed_mass) is ADOPTED unless look 3's MATCHED-STEPS read is INFERIOR (the upper one-sided 95 % bound of Δ < −δ = −3.5 pp). NON-INFERIOR adopts; NOT DETECTED without INFERIOR adopts ("take a risk"). The matched-wall-time deficit (cost, about −7 pp at s = +16.7 %) is ACCEPTED, to be recovered by the static-token rebuild and the performance work. The blob path is deleted at the adoption version break (§3.8). Purpose metric (1) is still tested if strength is NON-INFERIOR, and reported. | Waiting for look 3 to decide (costs a day of the refactor); requiring NON-INFERIOR strictly | owner 2026-10-06 |
| 2026-10-06 | **MATCHED-WALL-TIME definition fixed BEFORE look 3 is read** (survey `81e3a69a` finding 8: the paired benchmark used a 20-snapshot pool while the arms grew theirs from 0 to 5; in-arm steady-state cycles read +14.3 % ⇒ 13M by the benchmark's own rule, while in-arm END-TO-END wall, startup included, reads +16.9 % ⇒ 12M) | **Matched wall time = END-TO-END wall time, startup included**, measured within the arms: what a run actually costs. The checkpoint stays **12M**. The 13M steady-state read is reported beside it as a SENSITIVITY line, never as the read. Unchanged: the adoption rule is matched-steps only (owner, `079dee3e`), and wall time is reported. | Steady-state cycles only (excludes the 2× startup cost a real run pays) | orchestrator, delegated by the owner through 10-07 |
| 2026-10-06 | **The LOOK-3 CROSS plays at the TRAINING code, before any read** (`2d29c4c0` fixed a legality GIGO on the production input path: our active's request-order legality landed on the alphabetically sorted move slots by position, wrong on 6.8 % of real move decisions; no ARCH bump, so HEAD loads every X5 checkpoint and feeds it the CORRECTED input it never trained on) | The look-3 cross (and purpose metric (1), if run) executes from a checkout PINNED at **706fa536**: the look-3 seeds' own training pin; all 16 seeds trained on the pre-fix input. HEAD code is NOT used for any X5 read. The harness logs `git rev-parse HEAD` == 706fa536 and refuses otherwise. Looks 1–2 were played before the fix and are unaffected. | Playing at HEAD (every checkpoint read off its training distribution, the two arms possibly unequally) | orchestrator, delegated by the owner through 10-07 |
