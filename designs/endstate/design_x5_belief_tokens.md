# X5: discrete fixed-mass belief tokens + OTHER, design note and A/B pre-research

**Status: REVISED after independent review (2026-10-03); M2 / M3 open for owner brainstorm (§9). Nothing here is
built.** The first version (`00cdf0c2`, decisions `81578969`) was reviewed SOUND WITH FIXES with ten must-fix items
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
- **Masks are STRUCTURAL** (a count reaches zero), never a threshold on a continuous mass. OTHER's log-mass is computed
  as a logsumexp of the tail's log-presences (§3.2), so it is finite whenever its count is nonzero and needs no floor.

### 3.2 Mass: how π is built and updated within a battle

**Species, at every decision, from the current observation (no recurrent state):**
1. **Scores.** a_s = log P_T0(s | revealed) + δ_θ(s | ctx) over the valid set V = {species numbers ≥ 1} minus the
   revealed species. V is STRUCTURAL: the sentinel 0 and the revealed species are excluded exactly (π = 0, no logit),
   not by the prior's finite Species-Clause logit.
   - `P_T0` is the existing Smogon teammate naive Bayes. The prior stays SMOGON-ONLY (owner rule, 2026-08-15).
   - **δ_θ is new and state-dependent:** a Deep-Sets sum-pool (Zaheer et al. 2017) over the REVEALED opponent role
     tokens ⊕ the global token → a 2-layer MLP → species logits.
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
     EXCLUDED from the BCE (they carry no information). It is unreachable for species (n ≥ 382 against k ≤ 6) and for
     moves (hundreds of candidates against k_m ≤ 4); a deterministic test asserts the structural branch.
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
  - Its per-mon observation row comes from a static **dex-row table [n_species, 122]**, produced by THE observation
    encoder: the Rust `BattleVersion::encode` for "species s present, unrevealed set, full HP, no status".
  - An unseen gen-3 mon is pristine: it has never been on the field, so HP, status and boosts are exactly the defaults.
  - The table is a **committed artifact beside the model code**, not under `data/`, so a pinned run isolates it.
  - **Two gates (U1).** (i) A `sim`-tier gate regenerates the table with the encoder and requires byte equality.
    (ii) A **real-state cross-check**: over a fixed seeded set of real Rust battles, at each opponent mon's first
    appearance, its real encoded row must equal its dex row on every field except a DECLARED list of on-field fields
    (the active flag, an ability announced on entry, the field position). Any other differing field FAILS, naming it.
    This catches a synthetic state that the encoder renders differently from a real one.
  - No second encoder exists, and the runtime observation does not change.
  - A learned `hypothesis_marker` vector is added to the token (as E5's `tail_marker` is). The token-type table is not
    grown, because that would change every state_dict.
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

**The two invariances, scoped to class E.**
- **(I1) Zero = mask:** a hypothesis with π = 0 gives the same output as that seat masked. A presence of exactly 0 is
  the masked case, so there is no boundary.
- **(I2) Split = whole:** replacing one hypothesis (w) by two identical tokens (w/2, w/2) leaves every attention
  output unchanged (the ToMe identity). For the **flat pointer**, I2 reads: the probability of each EVENT (the sum
  over its copies) and of every other candidate is unchanged; the per-candidate output vector changes shape.
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
| the fixed-size construction | 64 bisection steps × [B, 388] sigmoid-and-sum for species, × [B, 400] for moves, no grad: ≈ 0.05 MFLOP/row of elementwise work; compile unrolls 64 steps (compile-time cost UNMEASURED, U2 measures it) | ESTIMATED |
| op elementwise | hypothesis defenders replace averaged ones (the same `[B,4,6]` shape); the **attacker gate opening adds per-hypothesis attacker rows**; M3 option (c) adds one averaged OTHER row, the cost the blob pays today | **NOT MEASURED** (the FLOP counter ignores elementwise work) |
| M3 option (a), + m hypothesis tokens | ≈ +0.65 MFLOP/row per token: m = 6 is ≈ +7 % forward, plus op rows ∝ the budget | ESTIMATED |
| `train_ms` | +0.3 to +0.6 s (+0.7–1.5 %) from the token; the op term unknown | ESTIMATED |
| T2 flush | ≤ +0.5 % (trainee only; opponents dispatch-bound) | ESTIMATED |
| memory | saved activations ≈ +1 %, ≈ +30 MB at micro-batch 2,048, against 2,218 MiB headroom | ESTIMATED |

**Pre-registered cost budget (build unit U8, GPU via `scripts/ops/gpu_lock.sh`, the learner benchmark).**
- **X5 `train_ms` ≤ +5 %, T2 flush ≤ +3 %, steady-state D-6 headroom ≥ 1,024 MiB at N = 256 with the X26 heads.**
- Over budget ⇒ STOP and report before any A/B GPU. The measured slowdown also enters §7's wall-clock rule.

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

### 3.8 Versioning, Rust, gates

**Flag.**
- Add **`--belief-tokens {blob,fixed_mass}`**, STRUCTURAL, default `blob` until the A/B rules.
- Bump `MODEL_CONFIG_VERSION` from the code's current value by one, with a `_migrate_config` default to `blob`. Add a
  `ModelFlag` row on all five surfaces, `requires` = `t0_species_prior`, `move_belief_mode`, `opp_intent`.
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
  - The bank's core input logs carry both teams, so N_ρ is exact. UNVERIFIED that the bank's reader exposes them; U7
    checks it first.
- **Both arms at ONE commit** through the flag, so the prober's arch-drift refusal does not apply. Forward passes only,
  on CPU.
- The bank's teams are pool teams. The owner's order is on-pool first; off-pool is a second column once a ladder bank
  exists (X8/X9).
- Its output never goes under `models/`.

Optional: a count DISTRIBUTION read with the nonrandomized PIT (Czado et al. 2009), if R1 flags a role.

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
  run (F-X5-4 / F-X5-5; the production fresh build is pinned to one thread since `50fdfdc2`, so this is a check); U8's cost budget passed. A run that breaks a precondition is
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
| G-1 | σ_run at 15M, the H2H run floor and snapshot jitter are unmeasured (P0 measures the last two). |
| G-2 | Negative evidence (an opponent NOT switching to X) is not modelled; X12. |
| G-3 | OTHER's embedding: `design_q_head.md` §10, logged by U8 (OTHER's attention share). The budget question is now §9 M3. |
| G-4 | That the Lane S bank exposes both full teams to a reader: UNVERIFIED (U7 checks first). |
| G-5 | PokaiTrainer (arXiv:2608.29197): its belief representation is unread. Read it before X4 returns. |
| G-6 | Whether the op's per-hypothesis attacker rows fit the elementwise budget: unmeasured (U8). |
| G-7 | The live head's on-pool sharpness, hence X5's real OTHER mass on-pool: UNVERIFIED (§3.3; a species-only proxy is a floor on sharpness). |
| G-8 | The CPU cost of one H2H cell (2,000 games, checkpoint against checkpoint): unmeasured (U0). |

### 8.3 Build plan (after the deletion pass, before X26)

Sizes are in agent-days. A "tier" is the gate a unit must pass before it lands. **Every unit is Opus-tier.**

| unit | what | size | tier / gates | agent |
|---|---|---|---|---|
| U0 | The checkpoint-vs-checkpoint mirrored H2H CROSS CLI (`sprt.py` + Rust eval core; the A/B's primary meter) + the P0 planning reads (§7.3) + its per-cell CPU cost | 1 | targeted + static; CPU under `mem_cap.sh` | opus-high |
| U1 | Dex-row table generator (Rust encoder) + committed artifact + `sim`-tier byte gate + the real-state cross-check (§3.4). (The `belief.rs` guard is DONE, `680edc36`.) | 1 | `sim` + cargo + static | opus-high |
| U2 | T0 hypothesis builder: δ_θ, the fixed-size construction (bisection, structural k = 0 / k = n, logsumexp OTHER), the single stable ordering, set BCE, BeliefHead re-target, moves; the `--belief-tokens` flag, versioning, registry; compile-time cost of the unrolled bisection | 2.5 | routine gate; tier contract; flag gates; the construction's tests | opus-high |
| U3 | Tokens into the chain: re-run the 41-site census on the built code; log-π bias in the transformer and every class-E pool (float masks, compile check first); class-M semantics per §9 M2; the op with hypothesis defenders and attackers, "alive" from `opp_addressable`; OTHER's physics per §9 M3; E5 owner bias; aux heads on hypothesis seats (§3.4); I1 / I2 and class-M / class-S tests per site | 3 | routine gate; invariance tests; obs golden untouched | opus-xhigh (GIGO risk; the orchestrator dispatches it) |
| U4 | Flat α pointer + OTHER labels; re-expressed cells with OTHER priced and `seat_live` consistent (§3.7); B ride-along re-base; `other_label_rate` | 2 | routine gate; `ridealong_update_test` bit-identity | opus-high |
| U6 | K9 golden: `init_group_sha256`, second entry, fp64 references, teeth test. (The thread pins are DONE: harness `0c25a1f4`, production fresh build `50fdfdc2`.) | 0.75 | routine gate | opus-high |
| U7 | Readers: `main.belief_roles` (R1–R4) + the intent / presence / OTHER purpose reads on the Lane S bank, per-run values for §7.4's across-seed inference | 1.25 | targeted + static | opus-high |
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
