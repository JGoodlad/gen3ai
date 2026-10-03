# X5: discrete fixed-mass belief tokens + OTHER, design note and A/B pre-research

**Status: DESIGN PROPOSAL, for owner review (2026-10-03). Nothing here is built.** This note is the build spec that
EXPERIMENT_BACKLOG row X5 asks for. It refines [`design_q_head.md`](design_q_head.md) §1 (the tokens) and §3 (the flat
opponent pointer). Where it departs from that spec, §3.9 says so and gives the reason; that spec carries a pointer here.
It also bundles the KL early stop (§5), plans the K9 golden re-bake (§6) and drafts the A/B (§7) for the orchestrator.

**Standard (owner, 2026-10-03).** "I don't want slop entering our core."
- Every design choice cites its literature, or says plainly that it is novel and names the risk.
- No homebrew where a grounded method exists.
- Every check is deterministic (standing rule 8).
- Open gaps are listed (§8.2), not hand-waved.

**Evidence tags.**
- MEASURED: banked, with a pointer.
- ESTIMATED: arithmetic on measured figures, with the assumption stated.
- UNVERIFIED: not checked.

Ledger lines are `L…` keys from [`../research_state/ledger_index.md`](../research_state/ledger_index.md).

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
   - Each unrevealed seat holds a real species hypothesis (top by inclusion probability), with presence weight π.
   - π is built from the Smogon prior ⊕ a learned, state-dependent delta.
   - π is turned into inclusion probabilities that sum EXACTLY to the unseen count, using capped
     probability-proportional-to-size (Tillé 2006). This replaces the spec's `min(1, k·q)`, which breaks the sum.
   - ONE new token, OTHER_species, holds the leftover mass. The opponent active's moves follow the same scheme, with
     mass 4 and OTHER_move re-using the active's existing E5 tail seat.
3. **Presence enters every reduction over opponent tokens as a log-weight attention bias.** This is ToMe's
   "proportional attention" (Bolya et al. 2023): softmax(l + log w) treats a token as w copies of itself. Two
   deterministic invariance tests pin it: zero presence equals masking, and splitting a token into two halves equals
   the token.
4. **Hypotheses get real physics.**
   - Each hypothesis seat's dex row is produced by THE observation encoder, as a committed artifact behind a parity
     gate. No second encoder exists, and the runtime observation does not change.
   - The op prices hypotheses as concrete defenders AND attackers.
5. **The opponent pointer becomes one flat list.**
   - Candidates: the move seats, OTHER_move, each revealed bench mon, each hypothesis, and OTHER_species.
   - A belief miss becomes an OTHER label instead of a masked row.
   - β and the content-addressed target are retired; the ride-along B head is re-based onto the flat list.
6. **Cost (§3.6).**
   - Sequence 61 → 62 tokens.
   - About +1.2 % forward matmul FLOPs and parameters roughly neutral (ESTIMATED).
   - The op's per-hypothesis attacker physics is NOT costed; build unit U8 measures it against a pre-registered budget.
   - No Rust runtime change. One label-writer guard.
7. **The KL early stop (§5) is NOT a rare safety net at `target_kl` 0.01.** On the sizing arms' per-epoch KL it would
   fire on at least 41–51 % of updates (MEASURED lower bound) and probably 61–74 %, leaving about 5.7–7.5 effective
   epochs. E5 vs E10 cost −10.6 pp on the untaught meter. This is an owner decision before the A/B (Q1).
8. **The replicate floor decomposes (§7.1) into about 98 % TRAINING-RUN variance and 2 % meter noise.**
   - σ_run ≈ 3.4 pp at 8M on the untaught meter. The 95 % CI is [1.5, ∞) from one degree of freedom; pooling gives
     about 3.0.
   - More games buy nothing. On that meter, a non-inferiority margin under 5 pp needs about 6–11 seeds per arm.
9. **Recommended A/B (§7.4).** A CPU pre-study (P0) on the banked sizing checkpoints measures whether a mirrored
   HEAD-TO-HEAD meter has a smaller run floor. If it does, the head-to-head is primary. Then 3 seeds per arm at 15M,
   about 15.4 GPU-h, with a pre-registered two-look extension to 5 seeds per arm (about 26 GPU-h).
   - Achievable margin (α = 0.05 one-sided, power 0.8): about 3.5 pp on the head-to-head path if its σ ≤ 1.5. On the
     untaught-meter path it is about 7 pp at 3 seeds and 5.4 pp at 5.
   - A slower X5 must also be non-inferior at matched wall-clock.
10. **Build (§8.3):** about 9–11 agent-days, after the deletion pass and before X26. Q1–Q6 (§9) are the owner's
    choices.

---

## 1. Problem statement

### 1.1 What X5 is

Backlog row X5, option 2: **discrete per-mon hypothesis tokens with FIXED total mass, plus an OTHER token**.
- **Their team:** revealed mons at weight 1, species hypotheses for the unseen remainder, and OTHER_species. The group
  mass is 6.
- **Their active's moves:** revealed moves pinned at 1, move hypotheses, and OTHER_move. The group mass is 4.
- Hypotheses are ALTERNATIVES weighted by presence, not teammates. OTHER's mass is the calibrated "how blind are we"
  quantity, so it is never renormalised away.

It adds no ply, so it stays in scope under the owner's one-ply rule. It is the North Star 1 retrain boundary: the X26
baseline launches on whichever architecture this A/B selects.

### 1.2 The blob as built (`900de5e2`, verified in code)

| piece | what it is today | pointer |
|---|---|---|
| **T0SpeciesPrior** | Parameter-free Smogon naive Bayes, `log P(s) + Σ_r log-lift(s, r)` over the chaos `Teammates`. Species Clause is a hard override. It produces ONE team-level categorical `[B, S]`: "the species in a hidden slot". It carries no count k and no without-replacement step. | `t0_species.py:42-118`, `belief_tables.py:498-540` |
| **BeliefSlots** | Each unrevealed opponent seat gets one of 6 learned per-position CONSTANT tokens, `unknown_slot_emb [6,128]`. No species information goes into the token. | `belief_heads.py:24-61` |
| **MoveBelief (hidden slot)** | E10 mixture `logit(P_T0 @ P(m\|s))` plus `move_head(token)`. The token is constant, so the learned part is a state-independent per-slot constant. | `belief_heads.py:265-420` |
| **Item / HP-type / spread (hidden slot)** | Read the constant token plus the species-0 prior row. They are state-independent, and they are never supervised there because labels exist only for revealed slots. | — |
| **E4 threat seats** | Top-6 of the opponent ACTIVE's candidate moves by sigmoid weight. The index is detached; `w` is differentiable. | `pointer_head.py:114-138`, `damage_op_blocks.py:936-962` |
| **E5 tail seats** | One per opponent mon, `[p_tail, worst_phys, worst_spec, revealed]` for the mass beyond rank 6. | `pointer_head.py:139-160` |
| **Damage op, hidden DEFENDER** | Priced on an AVERAGED defender: `E[def]`, `E[spd]`, `E[maxhp]` = `P_T0 @ tables`, applied before the nonlinear damage formula. Only the type multiplier is averaged per species. P(KO) is zeroed and full HP assumed. | `damage_op_blocks.py:351-422` |
| **Damage op, hidden ATTACKER** | Gated OFF (`att_gate` = revealed only): the op says nothing about what an unseen mon does to us. | `damage_op_pairwise.py:285` |
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

All entries were checked by the literature read of 2026-10-03, by abstract, PDF or source code. A row marked PARTIAL
had one claim not confirmed. "Adopt" means the design uses it; "reject" means it was considered and not taken, with the
reason.

### 2.1 Set and slot representations

| work | what it gives | for X5 |
|---|---|---|
| Zaheer et al. 2017, *Deep Sets* (NeurIPS; arXiv:1703.06114) | Permutation-invariant f = ρ(Σφ(x)) | **Adopt for the learned delta's input:** a sum-pool over the REVEALED opponent set (§3.2). It is also the formal description of today's averaged defender (a weighted sum-pool before a nonlinearity), which is what X5 replaces. |
| Lee et al. 2019, *Set Transformer* (ICML) — PARTIAL | Attention over sets; PMA pooling by learned seeds | No new block is needed: hypotheses join the existing self-attention. PMA is the description of our CLS pools. |
| Locatello et al. 2020, *Slot Attention* (NeurIPS) | Slots DISCOVER objects through competitive attention | **Reject.** Our entities are given and enumerable (species); there is nothing to bind. |
| **Bolya et al. 2023, *Token Merging* (ICLR; arXiv:2210.09461), Eq. 1** | "Proportional attention": `softmax(QKᵀ/√d + log s)`, where a token standing for s merged tokens gets the bias log s | **Adopt as THE citation for the log-presence bias.** The identity `softmax(l_i + log w_i) = w_i e^{l_i} / Σ_j w_j e^{l_j}` is elementary; ToMe is its published use for exactly "one token = w copies". |
| Ying et al. 2021 (Graphormer, NeurIPS); Press et al. 2022 (ALiBi, ICLR) | Additive attention-logit biases, learned (Graphormer) and fixed (ALiBi), train well | Precedent that a FIXED additive bias is benign, and the fallback if a learned scale on log w is ever needed (Q5). Our edge families are already this mechanism. |

### 2.2 Particle and mixture beliefs in POMDP RL

| work | for X5 |
|---|---|
| Ma et al. 2020, *DPFRL* (ICLR; arXiv:2002.09884) | **The closest precedent:** a weighted particle belief fed to the policy (through moment features). X5 feeds the same kind of object, a weighted discrete hypothesis set, through attention with a log-weight bias. Cite as related. |
| Ma et al. 2020, *PF-RNN* (AAAI); Karkus et al. 2018, *PF-Net* (CoRL) | Supports "a weighted set of discrete hypotheses beats one vector". We need no learned transition or resampling: our hypothesis space is enumerable and its prior is Smogon. |
| Igl et al. 2018, *DVRL* (ICML) | **Reject the machinery** (an SMC ELBO trains the belief). Our prior is known and our labels are exact (the true team), so plain supervision is the grounded choice. |
| Silver & Veness 2010, *POMCP* (NIPS); Hájek 1964; Chen, Dempster & Liu 1994 (Biometrika) | For the LATER search phase: conditional-Poisson (maximum-entropy, fixed-size) sampling turns X5's π into joint team worlds with the right marginals. Not part of X5. |

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
| Hu et al. 2021, *Learned Belief Search* (arXiv:2106.09086) | A SUPERVISED learned belief model replaces exact beliefs (> 60 % of the gain at 35× less compute). | **Adopt the training principle:** X5's delta is a supervised belief head on the true team (§3.2). |
| He et al. 2016, *DRON* (ICML) | An auxiliary opponent model inside the RL network. | Where α belongs: kept as an aux pointer under the existing gradient mode. |
| Perolat et al. 2022, *DeepNash* (Science) | Model-free equilibrium play with NO explicit belief. | The counter-position. Not adopted: our owner direction is discrete-first, and Q and search need an explicit opponent candidate list. |
| Wang 2024, MIT MEng thesis (`../references/wang2024_pokemon_rl.pdf`) | Random battles: MCTS samples ONE filling per trajectory from Showdown's random-team generator; the network sees "unknown" flags. | The generator prior does not exist in OU. Confirms the representation problem is open. |
| **Grigsby et al. 2025, *Metamon*** (arXiv:2504.04395; RLC 2025 per its README) | The observation holds ONLY the opponent's active mon. The team is inferred implicitly from memory over the trajectory. Replays are reconstructed with a human-team model. | **The main opposing design** (implicit belief). We keep explicit beliefs because Q and search need concrete opponent columns (`design_q_head.md` §3–§6). Metamon's team model is a precedent for a teammate-conditioned prior. |
| **Foul Play** (pmariglia; code at `6c467c0`, read) | Samples N determinized worlds at weight 1/N. Unrevealed mons are added ONE AT A TIME by mean teammate co-occurrence over the revealed mons (top 50). Smogon MOVE usage is treated as a per-mon INCLUSION rate (`1 − (1 − r)^{1/n}`). | Its teammate-conditioned prior is our T0. Its sampling does not respect the fixed set size; the author calls it "very non-scientific" for gen-3 OU. X5's fixed-mass π is the principled version. Move usage being an inclusion rate grounds §3.3. |
| Karten et al. 2025, *PokéChamp* (ICML) | Point estimates of unknown stats from usage; an LLM predicts opponent actions. | Point estimates, not a distribution. The LLM's action predictor parallels α. |
| Yu 2026, *PokaiTrainer* (arXiv:2608.29197) — PARTIAL | CFR over public belief states in VGC. | Its belief representation was not confirmed from the abstract. Read in full before X4's return (§8.2). |

### 2.5 Open sets, unseen mass, calibration

| work | for X5 |
|---|---|
| Scheirer et al. 2013 (TPAMI); Bendale & Boult 2016, *OpenMax* (CVPR) | Precedent for an explicit "unknown" bucket. Ours comes from a prior plus supervision, not from activations; borrow the idea only. |
| Good 1953 (Biometrika); Ferguson 1973 / Blackwell & MacQueen 1973 | The unseen-mass and "new table" mass literature. It supports OTHER's existence. X5 does not need a Good–Turing estimate, because the unseen COUNT is known exactly (6 − revealed) and OTHER is the in-model tail of π. |
| **Tillé 2006, *Sampling Algorithms*** (Springer); Särndal, Swensson & Wretman 1992 | **Adopt (§3.2):** π-proportional-to-size inclusion probabilities with the ITERATIVE cap. Set π = k·x/Σx; any π > 1 is fixed at 1 and the rest recomputed. The result has Σπ = k and π ≤ 1 exactly, which the spec's one-step `min(1, k·q)` does not. |
| Guo et al. 2017 (ICML); Gupta & Ramdas 2022 (ICLR) | ECE / reliability, with the multiclass-to-binary reduction: score each (species ∈ unseen set) indicator as a binary forecast. That reduction is X5's loss and its calibration read. |
| Brier 1950; Murphy 1973; Spiegelhalter 1986 (Stat. Med.) | Brier score with the reliability / resolution split, and the Spiegelhalter Z for calibration. **Adopt for presence indicators.** X5 should raise RESOLUTION over the blob at equal reliability. |
| Van Calster et al. 2016 (J. Clin. Epidemiol.) | The calibration hierarchy. A count test of Σ predicted against realised is MEAN calibration (level 1). **Adopt for role calibration (§4).** |
| Czado, Gneiting & Held 2009 (Biometrics) — content from memory | The nonrandomized PIT for count forecasts. Used only if a count DISTRIBUTION is read (§4.3). |

**Grounding check on F4, a fixed group mass.** The six team indicators are negatively correlated, because exactly six
are true. So the variance of a realised count is BELOW Σ p(1 − p), and a Z built on Σ p(1 − p) is conservative. §4
uses it knowing that. No reference for dependent-indicator count calibration was found, so the application is ours.

### 2.6 Evaluation methodology (§7) and PPO early stopping (§5)

- **RL evaluation across seeds.**
  - Henderson et al. 2018 (AAAI) and Colas, Sigaud & Oudeyer 2018 (arXiv:1806.08295; power analysis for seeds).
  - Agarwal et al. 2021 (NeurIPS; stratified bootstrap, IQM).
  - Patterson et al. 2024 (JMLR; empirical design, pre-registration).
  - **Bouthillier et al. 2021** (MLSys): randomise MORE variance sources per run, because seed-pinning one source
    hides variance.
- **Non-inferiority and sequential designs.**
  - Piaggio et al. 2012 (JAMA, CONSORT NI); FDA 2016 NI guidance; Schuirmann 1987 (TOST); Wellek 2010.
  - Pocock 1977; O'Brien & Fleming 1979; Jennison & Turnbull 2000.
  - Wald 1945 (SPRT). Van den Bergh's Fishtest notes on the pentanomial pair model are grey literature: the
    normalized-Elo note was verified, the GSPRT note was not.
- **Variance reduction.** Deng et al. 2013 (CUPED). Considered and **rejected** for the run-level term, because no
  PRE-treatment covariate exists: both arms differ from step 0. It is fine for meter-level noise, which is not the
  binding term (§7.1).
- **PPO KL early stop.**
  - Spinning Up: `kl > 1.5·target_kl`, checked per FULL-BATCH gradient step, default target 0.01 (code read).
  - SB3: the same threshold, checked per MINIBATCH before the step, on the `(r − 1) − log r` estimator (code read).
  - The 1.5 traces to the adaptive-KL rule in Schulman et al. 2017.
  - Andrychowicz et al. 2021 (ICLR) found KL penalties redundant under the PPO clip, and does NOT study early stopping.
  - **No paper evaluates the early stop itself.** It is an engineering default.

---

## 3. The design

### 3.1 Token set (production arm `--belief-tokens fixed_mass`)

| group | tokens | mass | revealed | hypotheses | OTHER |
|---|---|---|---|---|---|
| their team | the 6 existing opponent seats + **1 new seat, OTHER_species** | 6 | role token as today, weight 1 | the (6 − r) unrevealed seats hold the top-(6 − r) species by π, each a concrete mon token (§3.4) with weight π_s | k − Σ_top π = the tail mass; masked iff no unrevealed seat (r = 6) |
| their active's moves | the 6 E4 seats; **OTHER_move re-uses the active's E5 tail seat** | 4 | pinned at 1 | the top unrevealed moves by π_m | 4 − Σ seat weights; masked iff all 4 moves revealed |

- The sequence goes from 61 to **62 tokens**.
- The spec said "+2" (§3.9). OTHER_move takes the active's E5 seat because that seat already carries exactly the mass
  beyond the top-6. The bench mons' E5 seats are unchanged.
- **Ties in the top-k are broken by species (or move) number**, which is deterministic.
- **Masks are STRUCTURAL** (a count reaches zero), never a threshold on a continuous mass, so no rounding boundary
  exists (rule 8). A non-masked OTHER with tiny mass gets bias `log(max(mass, 1e-6))`, which is continuous.

### 3.2 Mass: how π is built and updated within a battle

**Species, at every decision, from the current observation (no recurrent state):**
1. `q = softmax(log P_T0(s | revealed) + δ_θ(s | ctx))` over species. The revealed species are excluded by Species
   Clause's hard override, as today.
   - `P_T0` is the existing Smogon teammate naive Bayes. The prior stays SMOGON-ONLY (owner rule, 2026-08-15).
   - **δ_θ is new and state-dependent:** a Deep-Sets sum-pool (Zaheer et al. 2017) over the REVEALED opponent role
     tokens ⊕ the global token → a 2-layer MLP → species logits.
   - The last layer is zero-initialised, so at a cold start q = the Smogon prior exactly.
   - It runs in tier T0, on pre-transformer tokens only (`tier_contract`).
   - It replaces the constant `unknown_slot_emb`, so the hidden-slot learned delta finally depends on state (§1.3).
2. `π = πps_capped(k = 6 − r, size = q)` (Tillé 2006): set π = k·q; while some π > 1, fix it at 1 and rescale the
   rest to the remaining count.
   - It converges in ≤ k passes, and the pass count is fixed at k for a static graph.
   - It is differentiable almost everywhere.
   - **Σπ = k and 0 ≤ π ≤ 1 hold exactly.** That makes the group mass 6 a property of the construction, not a learned
     tendency.
3. The hypotheses are the top-k species by π. **OTHER_species = Σ_{tail} π_s** = k − Σ_top π.

**Moves (the opponent active):**
- The MoveBelief per-move presences, i.e. Smogon usage, which is an INCLUSION rate (Foul Play's treatment; §2.4), ⊕
  the existing learned head, go through the same `πps_capped(k_m = 4 − r_m, size = p)`.
- Revealed moves stay pinned. The E4 seats are the top unrevealed moves by π_m; OTHER_move takes the rest.
- Hidden Power keeps its typed composition (`compose_typed_hp`) before the πps step.

**Within-battle updating.**
- Every reveal changes r (or r_m) and the conditioning set, and π is recomputed at the next decision: the same
  "re-condition on each reveal" as `design_q_head.md` §1.
- Not modelled: NEGATIVE evidence, such as "they did not switch to X when it was the obvious answer". That needs the
  learned roster mixtures (X12) and is recorded as a gap (§8.2).

**Supervision.**
- Full-support BCE on π against the true unseen set: one binary indicator per species. This is the multiclass-to-binary
  reduction (Gupta & Ramdas 2022), a proper score per indicator.
- It needs NO slot matching. Unseen slots are exchangeable in gen 3 (no team preview), so the Hungarian step goes.
- **OTHER's calibration then follows by linearity:** E[number of true unseen species outside the list] = Σ_tail π
  when π is calibrated.
- Labels: the existing `belief_species [6]` with the revealed mask. **No new Rust label.**
- **BeliefHead is kept as the post-transformer trunk-shaping aux**, re-targeted to the same set-level BCE, so the arm
  changes the representation and not the shaping signal (one lever).
- The coefficient stays at 0.05; `belief_grad_mode` stays `shaping`.

### 3.3 What OTHER is

- OTHER is an ENTITY, not a scalar.
- **Its embedding** is a learned "rest" vector plus a linear map of the π-weighted mean SPECIES embedding over the
  next 32 tail candidates (the move token likewise). This is the one place averaging is right, because OTHER *is* the
  marginal.
- **OTHER gets no physics.** The op and edge rows for it are zero, with an `is_other` flag. Giving it averaged physics
  would rebuild the blob inside the new representation.
- It is a key in attention with its log-mass bias, a pointer candidate, and (when X4 returns) a Q column.
- Rejected alternative: OTHER as the op's expectation over the whole tail. Its cost scales with the tail size, and it
  is the blob again.
- OTHER's embedding is still an open question (`design_q_head.md` §10). The build ships learned + tail-mean and logs
  OTHER's attention share.

### 3.4 How hypothesis tokens are built, and the physics they get

- **Content.**
  - A hypothesis seat is a CONCRETE mon of species s, encoded by the SAME `pokemon_encoder` as a revealed one.
  - Its per-mon observation row comes from a static **dex-row table [n_species, 122]**, produced by THE observation
    encoder: the Rust `BattleVersion::encode` for "species s present, unrevealed set, full HP, no status".
  - An unseen gen-3 mon is pristine: it has never been on the field, so HP, status and boosts are exactly the defaults.
  - The table is a **committed artifact beside the model code**, not under `data/`, so a pinned run isolates it.
  - A `sim`-tier gate regenerates it with the encoder and requires byte equality. No second encoder exists, and the
    runtime observation does not change.
  - A learned `hypothesis_marker` vector is added to the token (as E5's `tail_marker` is). The token-type table is not
    grown, because that would change every state_dict.
  - MoveBelief, ItemBelief, HP-type and spread then read a SPECIES-SPECIFIC token for hypotheses, through the existing
    per-species Smogon priors. The hidden slot stops being a constant.
- **Physics.**
  - The damage op prices each hypothesis seat as a concrete DEFENDER: its own stats, no averaging. It also prices it
    as an ATTACKER: the gate opens for hypotheses, using its per-species move posterior.
  - Each op row is then weighted by presence wherever it is reduced (§3.5). Effect: E[f(x)] replaces f(E[x]).
  - The attacker side is new information to the network: "what can their unseen mons do to me".
- **Rejected:** a torch-side builder of synthetic observation rows. That is a second encoder and a drift risk. The
  synthetic-key registry exists because two encoders of one row drifted before.

### 3.5 How tokens enter the phase chain, and what each head reads

| tier | change |
|---|---|
| T0 RESOLVE | δ_θ, πps, hypothesis selection, dex-row gather, `pokemon_encoder` on hypothesis rows, OTHER embeddings. `BeliefSlots` is retired in this arm. |
| T1 REASON | The op prices hypotheses (defender + attacker). Edge cells on hypothesis seats as on revealed ones; OTHER's are zero. The `TeamTransformer` bias adds **`log w_j` on every query's logit for key j, in all heads**, for opponent seats, OTHER and E4/OTHER_move. Revealed keys get 0, masked keys −1e9 (the key-padding addend that already exists). |
| T2 DECIDE | `cls_pool` (`their_cls`, `value_cls`), `HiddenOppBeliefPool` and `value_entity_pool` get the SAME log-w key bias. The flat α pointer (§3.7) replaces `alpha_head` + `beta_head`. Intent / pair / switch-branch / conditional-threat cells read α through §3.7's re-expression. |
| T3 DELIVER | Unchanged. |

**The weighting rule, enforced, not trusted.** Every reduction over opponent tokens must be presence-aware. The build
enumerates each one and pins two invariances with deterministic tests on fixed inputs, exact in fp64 and to 1e-6 in
fp32. A presence at exactly 0 is the masked case, so there is no boundary.
- **(I1) Zero = mask:** a hypothesis with π = 0 gives the same output as that seat masked.
- **(I2) Split = whole:** replacing one hypothesis (w) by two identical tokens (w/2, w/2) leaves every attention
  output unchanged (the ToMe identity).

A reduction that fails either test is a bug: a token that "counts as a whole mon".

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
| op elementwise | hypothesis defenders replace averaged ones (the same `[B,4,6]` shape); the **attacker gate opening adds per-hypothesis attacker rows** | **NOT MEASURED** (the FLOP counter ignores elementwise work) |
| `train_ms` | +0.3 to +0.6 s (+0.7–1.5 %) from the token; the op term unknown | ESTIMATED |
| T2 flush | ≤ +0.5 % (trainee only; opponents dispatch-bound) | ESTIMATED |
| memory | saved activations ≈ +1 %, ≈ +30 MB at micro-batch 2,048, against 2,218 MiB headroom | ESTIMATED |

**Pre-registered cost budget (build unit U8, GPU via `scripts/ops/gpu_lock.sh`, the learner benchmark).**
- **X5 `train_ms` ≤ +5 %, T2 flush ≤ +3 %, steady-state D-6 headroom ≥ 1,024 MiB at N = 256 with the X26 heads.**
- Over budget ⇒ STOP and report before any A/B GPU. The measured slowdown also enters §7's wall-clock rule.

### 3.7 The opponent pointer, and the A and B heads re-based

**One flat candidate list** (`design_q_head.md` §3):
- the active's 6 move seats + OTHER_move;
- switch → each revealed bench mon;
- switch → each hypothesis seat;
- OTHER_species.

**Scoring and labels.**
- One shared scorer over candidate tokens, plus the candidate's `log w` as a logit bias, then one softmax.
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
- A cell that needs a property of the switch target (for example P(Ghost) for Rapid Spin) takes OTHER's value from the
  tail marginal.

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
- Bump `MODEL_CONFIG_VERSION` 134 → 135, with a `_migrate_config` default to `blob`. Add a `ModelFlag` row on all five
  surfaces, `requires` = `t0_species_prior`, `move_belief_mode`, `opp_intent`.
- **No `ARCH_SIGNATURE` bump while both arms must build at one commit.** The bump comes with the loser's deletion,
  through `snapshot._DEAD_FEK_*`, as the P11b deletions do.

**Rust.** No runtime observation or label change.
- One label-writer guard: `labels/belief.rs:79-81` silently drops a species with no dex number, which would undercount
  an OTHER label. Make it throw (F-X5-3).
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
- The new invariance tests I1/I2 (§3.5), the πps sum tests, and the dex-row parity gate.
- On adoption: the mirror sync, `mode_flag_doc_gate`, `arch_tables_test`, the K9 golden (§6), ARCHITECTURE.md and
  CHANGELOG in the same commit.

### 3.9 Departures from `design_q_head.md` §1 / §3, and why

| spec | this note | why |
|---|---|---|
| presence = `min(1, k·q)`, OTHER = k − Σ hypotheses | iterative capped πps (Tillé 2006) | `min` loses mass whenever a weight is capped. The lost mass lands in OTHER and reads as "unnamed species" when it is really an over-confident top hypothesis. Capped πps keeps Σ = k exactly. |
| "≈ 29 → ≈ 31 tokens", +2 | 61 → 62, +1 | The production sequence is 61 (event seats, measured). OTHER_move re-uses the active's E5 seat, which already holds the beyond-top-6 mass. |
| OTHER embedding learned + out-of-list average | the same, **no physics rows for OTHER** | Averaged physics for OTHER would re-create the blob. |
| (unspecified) the learned delta's input | Deep-Sets pool over REVEALED tokens + global, at T0, zero-init | The tier order forbids reading refined tokens at T0. The zero-init keeps cold start = the Smogon prior. |
| (unspecified) belief loss | full-support set BCE; no Hungarian matching | Exchangeable slots; a proper per-indicator score; OTHER calibrated by linearity. |

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
7. **A larger hypothesis budget** (7–8 + OTHER, mass 6). Deferred (`design_q_head.md` §10). U8's OTHER-mass read early
   in games decides whether the budget binds.
8. **Hand-defined ROLE tokens.** Rejected by the owner (2026-10-02): roles must be learned.
9. **A torch-side synthetic-row builder.** A second encoder (§3.4).

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

M_ρ = Σ_revealed P̂_i(m) + Σ_hyp w_h·P̂_h(m) + w_OTHER·P̄_tail(m)

- P̂ is the network's own move posterior. Revealed moves are pinned at 1.
- **For the blob arm**, the hypothesis terms are replaced by its hidden-slot move presences, Σ_hidden P̂_slot(m). That
  is the blob's own belief about m among unseen mons.
- **For the prior column**, T0 → πps → Smogon P(m | s), so the learned deltas' contribution is separable.
- N_ρ is the number of opponent team members whose TRUE moveset holds m.

The four reads:
- **R1, mean calibration** (Van Calster et al. 2016, level 1). Δ_ρ = mean(M_ρ − N_ρ), stratified by reveals r = 0…5,
  with a battle-clustered bootstrap 95 % CI (Agarwal et al. 2021's stratified bootstrap). Summary: the usage-weighted
  mean |Δ_ρ| over roles.
- **R2, count Z.** Σ(N − M) / √Σ Var, with Var = Σ p(1 − p): CONSERVATIVE under the fixed group mass (§2.5).
  Reported, and never the sole verdict.
- **R3, substitute double-counting.**
  - A substitute pair (s₁, s₂) is pre-registered from Smogon only: both have P(m | s) ≥ 0.5 for a shared role m, and
    their teammate lift is < 1, i.e. they co-occur less than chance.
  - Rows: decisions where exactly one of the pair is in the true unseen set and neither is revealed.
  - Read: the rate at which BOTH have π ≥ 0.25 (π for X5; the πps of BeliefHead's posterior for the blob, so the two
    are comparable). Also the continuous excess π₁ + π₂ − 1.
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

## 5. The KL early stop (bundled by the owner, 2026-10-02)

### 5.1 What the vendored stop does (`instrumented_ppo/ppo.py:541-556`)

- It checks ONE MICRO-BATCH's approx-KL, (r − 1) − log r over 2,048 rows, at the current parameters before that
  micro-batch's backward pass, against 1.5 × `target_kl`.
- At the production shape, 98,304 rows ÷ 2,048 = 48 micro-batches per epoch. With accumulation 32, each epoch is one
  full group plus one ragged group: **2 optimizer steps per epoch, 20 per update at E10**. `main.dose` counts the same
  way.
- A trip zeroes the open accumulation group and skips every remaining epoch. The critic, belief and intent losses share
  that loop, so they lose those passes too.
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

How to read the table:
- The lower bound is MEASURED and exact on the open-loop, no-feedback replay. No epoch mean lay within 1e-6 of 0.015.
- The model is ESTIMATED: gamma-distributed micro-batch draws around the reconstructed KL path, with coefficient of
  variation c/√2048. c, the per-row coefficient of variation, has a plausible range of 4–7 from the age-1 probe.
- **KL is CONCAVE in steps, not quadratic:** epoch 0's first step already reaches 34–49 % of the final epoch's KL.

**Consequence at B with c = 4–7.**
- The stop keeps about 5.7–6.5 effective epochs, i.e. 57–65 % of the nominal dose.
- O8 found E5 vs E10 costs **−10.56 pp U [−13.12, −7.94]**, at about 60 % of B's dose (X27 notes the dose mismatch).
- So a strength cost of that order **cannot be ruled out**. It is probably smaller, because the stop cuts high-KL steps
  where the clip already zeroes most of the policy gradient. But the value and belief heads lose those passes outright.

### 5.3 Wiring

- Read `target_kl` from `recipe_surface.kl_controller()`. Set `model.target_kl` in `apply_training_hparams`
  (`model_build.py:94`), which both build paths call, so a fork of a pre-stop checkpoint does not restore `None` from
  the zip.
- Record it in `metadata.json` and the dose block.
- **Log per update:**
  - `train/kl_stop` (0/1) and `train/kl_stop_count` (cumulative);
  - `train/kl_stop_epoch` and `train/kl_stop_micro`;
  - `train/kl_stop_value`;
  - `train/opt_steps_realized` and `train/epochs_completed`.
- **Fix three couplings in the same unit:**
  - **Feed the controller a stop-aware statistic.** It reads `train/approx_kl` = the LAST epoch's mean. After an
    epoch-0 trip that reads ≤ 0.001 and pushes the LR UP, a positive feedback loop. The first step alone overshot 0.015
    on 9/86 (A2) to 14/81 (C) updates, clustered.
  - `main.dose` must count REALIZED steps, not nominal epochs (`dose.py:149-175`).
  - `update_fit.dry_update` runs with `target_kl = None`, so the startup memory fit is not cut short.
- **Tests:**
  - An accumulation-aware stop test on the golden learner with `target_kl = 0.0008`. It trips deterministically at
    epoch 1, micro-batch 3: KL 0.00342 against threshold 0.0012, far from the boundary.
  - The K9 golden itself is byte-IDENTICAL with `target_kl` 0.01, because its largest micro-batch KL is 0.00342. The
    golden does not exercise the stop. Move `golden_recipe`'s `target_kl` to the recipe source and re-record; the
    history row should read "changed: []".

### 5.4 The choice the owner has to make (Q1)

- **(a) The stop as an ACTIVE trust region at 1.5 × 0.01.** As bundled. It is a recipe change of DOSE magnitude, and
  its own strength effect is unmeasured. Both A/B arms carry it, so the X5 comparison stays valid. Neither arm is then
  comparable to any pre-stop run (sizing B, N0).
- **(b) The stop as a RARE SAFETY GUARD.** Set the stop threshold so that ≤ 5 % of the sizing updates would trip on the
  lower-bound read, decided by a 10-minute CPU read of the banked per-epoch KL before the build. It keeps the
  controller's band at 0.01.
- **(c) Check the KL on the accumulation-GROUP mean** (65,536 rows, the Spinning Up form) instead of per micro-batch.
  This removes the 2,048-row estimator noise. It lands near the lower bound: about 52 % fire rate at B, 15.0 of 20
  steps.

**Recommendation: (b) + (c) for the X5 A/B.**
- The X5 verdict should not ride on an unmeasured recipe change of the size that cost 10.6 pp in O8.
- (a) then becomes its own lever, after X26, with its own arm.
- This is the owner's call: the bundling decision was his.

---

## 6. The K9 learner golden re-bake

**What it pins today** (`training/learner_golden.py`, `learner_golden_test.py`, `designs/training/learner_gates.md`).
- Inputs:
  - a production-surface learner rebuilt from seeds;
  - a committed 64-row real Rust buffer (`learner_golden_buffer.npz`, sha `c6008684…`);
  - the golden recipe (2 epochs, micro-batch 16 × accumulation 3, LR 2.8e-5);
  - one eager fp32 CPU `train()` on 1 thread.
- Compared: initial and post-update parameter sha256, 41 per-group post-update hashes and 19 scalars, all EXACT.
- Keyed by torch version only, not by arch signature.
- Re-baked with `python -m agents.training.learner_golden record --reason …`, which appends a history row.

**Re-bake plan, and what proves it CORRECT rather than merely new:**
1. **X5 OFF is byte-identical.** With `--belief-tokens blob` (the default), the EXISTING golden passes unchanged at the
   X5 commit. This proves the flag constructs nothing and draws no global RNG when off. Land the KL-stop re-record (§5)
   as its own byte-neutral commit FIRST, so the two changes never share a re-bake.
2. **X5 ON** gets a SECOND golden entry (key `fixed_mass`), built through `build_learner(args = production + X5)`.
   Before it is recorded, these must hold:
   - losses, KL and gradient norm finite (K9(c));
   - KL and clip fraction in range;
   - **two processes with different `PYTHONHASHSEED`, both at 1 thread, produce identical hashes**;
   - the NON-X5 parameter groups' INITIAL bytes equal the blob arm's, if X5 initialises from an isolated generator.
     That needs the harness gap closed: store `init_group_sha256`.
3. **Independent references (fp64 numpy, on the golden's 64 rows):**
   - Σ team mass = 6 and Σ move mass = 4 exactly;
   - πps against a direct Tillé implementation;
   - OTHER's bias is −1e9 iff its count is 0;
   - attention with log-w bias equals the w-weighted renormalised softmax;
   - I1 / I2 (§3.5).
4. **Not vacuous:**
   - every X5 parameter group's post-update hash differs from its initial hash;
   - each new loss key is logged and nonzero;
   - a TEETH test: perturb k or the OTHER mask, the golden must FAIL naming X5's groups; revert, and it passes.
   - Coverage: the buffer holds 29/64 rows with all six opponent species known, but only **2/64** with an opponent mon
     with 4 revealed moves, so the OTHER_move-masked case is thin. Assert per-case counts ≥ 2, or rebuild the buffer
     (`rebuild-buffer --reason`, which invalidates every entry).
5. **Harness hazard to fix first (F-X5-4).** A learner built outside `_one_thread` at 8 threads moves the INITIAL hash.
   The test passes only because `conftest.py` pins threads. `build_learner` must enter `_one_thread` itself, or a
   re-bake from a CLI shell records a different init.

Not covered by the golden: compiled and CUDA numerics. Those belong to K8's regions, startup parity and the canary.
X5 makes no observation change, so the obs golden must stay green untouched.

---

## 7. The A/B design (the orchestrator decides the threshold; this is the draft)

### 7.1 Decomposing the 4.9 pp replicate floor

The script is at `/home/goodlad/.claude/jobs/9c36ca35/tmp/x5/vardecomp.py`. It reads the banked
`../research_state/measurements/m5_sizing/results/meters_read.json`.

**The model.** For two runs' untaught-meter (U) levels, E[Δ²] = 2σ²_run + σ²_meter,Δ. σ²_meter,Δ comes from the
meter's own team-clustered, paired CI, which contains game and team noise only. The meter is deterministic at seed 0,
concurrency 1, and has reproduced exactly five or more times (ledger L20461 and later). So everything the CI omits is
RUN-level, which is UNDERSTANDING rule 19's distinction.

| pair | depth | games | Δ (pp) | meter SE(Δ) | meter share of Δ² | σ_run (pp) |
|---|---|---|---|---|---|---|
| **A′ − A2** (seed 1002 vs 1001, same pin, M5 core, opponent N0@24M) | 8.06M | 600 / team | −4.90 [−6.21, −3.60] | 0.66 | **1.8 %** | **3.43**, 95 % CI [1.53, ∞) (1 df) |
| W_b − W (old lineage, opponent `ai_v9_29@24M`, L19064) | 75M | 200 / team | 3.69 | 1.40 | 14 % | 2.41 (1 df) |
| B − A2 (same seed, N 256 vs 48; sensitivity only, assumes no N effect) | 8M | 600 / team | −2.71 [−4.50, −0.94] | 0.91 | 11 % | 1.80 |
| pooled (first two) / pooled (all three) | — | — | — | — | — | **2.97** (2 df, 95 % CI [1.54, 18.6]) / 2.64 |

**Facts in the replicate pair.**
- All 8 teams moved the SAME way (per-team Δ −2.2 to −8.0, SD 2.0). That fits a run-level strength shift, not team
  noise.
- **Guard meter (G-A) on the same pair: Δ −0.83 [−4.79, +3.13].** Its meter variance (SE 2.0) exceeds the observed Δ²,
  so G-A's run component is undetectable at 1,200 games. A general shift of U's size would have read about −4.8 on G-A
  (≈ 2 SE).
  - That is weak evidence that **part of U's run variance is specific to its one opponent** (a seed × N0 interaction)
    rather than general strength.
  - It motivates P0 (§7.3).

**Verdict.**
- **Training-run variance dominates: about 98 % of the replicate gap at 8M and 600 games per team; meter noise is
  about 2 %.**
- More games, a GSPRT on games, or CUPED all reduce the 2 %, and buy nothing.
- The cheap remedies are only:
  - (i) a meter whose RUN floor is smaller;
  - (ii) averaging out within-run snapshot jitter, if that is a material share of σ_run (unmeasured);
  - (iii) more seeds, which is the expensive remedy.

**Caveats.**
- Each σ estimate rests on 1–2 df. UNDERSTANDING rule 3 calls a floor "the MAX pairwise |Δ|", and a single pair
  "bounds a floor, does not estimate one".
- The two U floors are against DIFFERENT opponents (F-X5-5).
- No untaught read exists at 15M or 25M on any lineage, so σ_run at the A/B's depth is UNMEASURED.
- 8M 3.43 against 75M 2.41 suggests at most a modest fall with depth.

**FINDING F-X5-6.** The sizing study and the learner battery ruled non-inferiority with "95 % CI lower bound > −3.69",
where the CI was the meter's game-and-team CI. That CI omits the run term. Their nominal α was not their real α, and
the sizing registration already flagged "n = 1 is not decisive". This design corrects it by putting σ_run in the SE.

### 7.2 Sizing: margin by seeds, depth and meter

With known σ, true Δ = 0, one-sided α and power 1 − β, the margin is δ = (z_{1−α} + z_{1−β})·√(2(σ²_run + σ²_m)/K),
where K is seeds per arm and σ²_m = 0.22 pp² per run at 600 games per team. All values ESTIMATED from §7.1.

**Untaught meter (U), α = 0.05 one-sided, power 0.8, δ (pp):**

| σ_run | K = 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 2.5 | 8.9 | 6.3 | 5.2 | 4.5 | 4.0 | 3.7 | 3.2 | 2.8 |
| 3.0 (pooled) | 10.7 | 7.6 | 6.2 | 5.3 | 4.8 | 4.4 | 3.8 | 3.4 |
| **3.43 (8M pair)** | 12.2 | 8.6 | **7.0** | 6.1 | **5.4** | 5.0 | 4.3 | 3.8 |

- At α = 0.025 (the CONSORT convention, a two-sided 95 % CI), add about 13 %: 7.9 at K = 3 and 6.1 at K = 5 for
  σ = 3.43.
- With σ estimated INSIDE the experiment (t-test, df = 2(K − 1)), the margin at K = 3 is 8.7, because t with 4 df is
  costly. This is why §7.4 uses a pre-registered σ.
- **Seeds per arm for a target margin** (σ_run 2.5 / 3.0 / 3.43):
  - δ = 3.69 needs 6 / 9 / 11;
  - δ = 5.0 needs 4 / 5 / 6.

**Head-to-head (H2H) meter, if its run SD is σ_h.** Every X5 seed plays every blob seed on mirrored pairs (a K × K
cross); the mean has variance 2σ_h²/K + meter. At K = 3:

| σ_h | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|
| δ | 2.7 | 3.5 | 4.4 | 5.4 |

σ_h is UNMEASURED. P0 measures it.

**GPU-hours** (sequential, one GPU, N = 256):
- blob at 15M ≈ 2.5 h (brief), X5 ≈ 2.55–2.6 h (ESTIMATED from §3.6, op term pending U8);
- K = 3: ≈ 15.3 GPU-h;
- K = 5: ≈ 25.5;
- K = 8: ≈ 41.

**15M vs 25M.**
- 25M costs 1.67× per run. It beats 15M at equal GPU-h only if σ_run(25M) < 0.77·σ_run(15M), i.e. 3 seeds at 25M
  against 5 at 15M.
- The only depth evidence (3.43 at 8M against 2.41 at 75M, 1 df each) does not show that.
- **Recommend 15M.**
- Risk named: if X5's benefit appears only late (beliefs matter more as play sharpens), 15M under-reads it. This is a
  non-inferiority design, so that risk falls on the adoption side, not the safety side.

**Meter choice.**
- U: the established meter with a measured run floor, but it has ONE opponent.
- H2H: directly answers "is X5 at least as strong as blob". It is non-transitivity-prone (`design_league_decisions.md`
  §0), but between two arms of one recipe that is acceptable as a GUARD-type question. Its run floor is unknown.
- An outside panel (SmallRL, Kakuna, Foul Play): too slow per game for the needed precision, and its run floor was
  meter-dominated at 100 games (F-X5-7).
- **A GSPRT on one seed pair answers "is THIS X5 run better than THIS blob run"**, which is a meter-noise question. It
  is the right per-cell meter inside the cross, NOT the population inference.

### 7.3 P0, a CPU pre-study on banked checkpoints (before any A/B GPU)

**Question.** Is the run floor smaller on a mirrored H2H than on U, and how big is within-run snapshot jitter?

**Inputs.** `sizing_A2_n48_e10_s1001`, `sizing_Ap_n48_e10_s1002`, `sizing_B_n256_e10_s1001` (read-only), their finals,
and their 8.0M checkpoints, which are about one update from final.

**Reads, all CPU:**
1. **H2H round-robin of the three finals,** mirrored pairs on the Rust eval core, 1,000 pairs per edge (meter SE
   ≈ 1.1 pp).
   - A2 vs A′ is the pure seed edge. The two B edges are sensitivity edges, assuming no N effect.
   - σ_h = |Δ_A2,A′| / √2, from 1 df plus 2 sensitivity.
2. **U on the 8.0M checkpoints of A2 and A′** (600 per team). U(final) − U(8.0M) is snapshot JITTER at about one update
   of separation.

**Tool.** No CLI plays checkpoint against checkpoint offline today: `main.anchors` takes external opponents only. U0
builds one from `sprt.py` + the Rust eval core (about 0.5 agent-day).

**Pre-registered rule (P0 → the A/B's primary meter).**
- **H2H is primary iff σ_h ≤ 1.5 pp.** Otherwise U is primary and H2H is secondary.
- If |U jitter| ≥ 1.5 pp on either run, U is read as the MEAN of the last three 1M-spaced checkpoints.

### 7.4 Recommended design and pre-registered decision rule

**Arms and order.**
- Two arms, blob and fixed_mass, at ONE commit through the flag. Production recipe at N = 256, with the §5 stop (per
  Q1) and the X26 ride-along heads ON in both (§9 Q4).
- Seeds 1001, 1002, 1003, the SAME seed ids across arms, so team and opponent schedules are shared. Pairing is for
  blocking only; sizing assumes zero correlation.
- Checkpoints every 1M from 12M.
- Arm order ABBA by seed (X1 B1 B2 X2 X3 B3), to block box drift.
- Through the training agent, `gpu_lock` throughout.

**Stage 1: K = 3 per arm, 15M** (≈ 15.3 GPU-h). Primary meter per P0.

**Variance used.** σ̂ = max(pre-registered prior, the in-experiment estimate). The prior is σ_run 3.43 for U, or P0's
σ_h for H2H.
- A within-arm spread larger than the prior's 95 % predictive bound makes the read INCONCLUSIVE, and it goes to stage 2.
- This keeps the decision deterministic and conservative.

**Statistic.** Z = (Δ̂ + δ) / SE, with Δ = X5 − blob in pp and SE = √(2(σ̂² + σ²_m)/K).

**Two-look O'Brien–Fleming boundaries** (information fraction 3/5, one-sided α = 0.05): **look 1 z ≥ 2.188, look 2
z ≥ 1.695**. At α = 0.025: 2.572 and 1.992.

| outcome at a look | rule |
|---|---|
| **NON-INFERIOR** | Z ≥ the look's boundary |
| **INFERIOR** | the upper 90 % bound of Δ < −δ |
| look 1, neither | stage 2: +2 seeds per arm (1004, 1005; ≈ +10.2 GPU-h), then the final look |
| look 2, neither | **NOT DETECTED.** Blob stays; X5 goes back to the owner with the read. Never "equivalent" (rule 6). |

**Margins (owner / orchestrator choice, Q2).**
- H2H path (σ_h ≤ 1.5): **δ = 3.5 pp** at K = 3.
- U path: δ = 7.0 at K = 3, 5.5 after the K = 5 extension (σ 3.43). A tighter U margin costs K = 8 (4.3 pp, ≈ 41 GPU-h)
  or more.

**Purpose metrics (fixed-sequence: tested only after strength is NON-INFERIOR, so the family α holds).** Read on the
Lane S bank, on-pool primary, CI clustered by battle:
1. Opponent-intent log loss on the common event space (the realised action's probability; a blob miss is reported as
   its own column, never floored into the score).
2. Species presence Brier (resolution).
3. OTHER calibration (R4).
4. Role calibration R1 / R3.

X5 is **ADOPTED** iff strength is NON-INFERIOR and (1) improves with its CI excluding 0, while (2)–(4) are reported.

**Speed rule (X5 non-inferior but slower).** Slowdown s = the median update-cycle ratio, X5 / blob, across the arms'
updates outside eval cycles.
- **s ≤ 5 %:** adopt on the rule above.
- **s > 5 %:** ALSO require non-inferiority at MATCHED WALL-CLOCK. Compare blob at 15M against X5 at the checkpoint
  ≤ 15M/(1 + s) (the nearest 1M checkpoint at or below, so it is deterministic). This is the
  strength-per-GPU-hour rule of `design_q_head.md` §8.
- **s > 15 %:** stop and report before any meter read, as an optimisation unit.

**Deletion.** On adoption, the blob path is deleted in the next deletion unit, and the signature is bumped (§3.8).
The X26 baseline launches on `fixed_mass` (§9 Q4).

---

## 8. Risks, open gaps, build plan

### 8.1 Risks

| risk | mitigation |
|---|---|
| The KL stop is a dose-sized recipe change riding on the A/B | Q1; recommendation (b) + (c) |
| A reduction that does not weight by presence ("counts as a whole mon") | I1 / I2 invariance tests per reduction (§3.5) |
| The hypothesis attacker op adds unbudgeted cost | U8's pre-registered budget, stop on breach |
| δ_θ memorises the pool | The owner's rule: a success milestone. On-pool first, with off-pool reported at the X8/X9 banks. The Smogon prior is never replaced, only ⊕'d |
| The bundle is two changes (tokens + flat pointer), so a strength regression cannot be attributed | The flat pointer needs the hypothesis candidates. α stays `detached` (it cannot move the policy's trunk), so its effect on strength is only through the re-expressed cells. If NOT DETECTED, a cheap follow-up ablates the cell inputs to blob-equivalent |
| A mon with fewer than 4 moves breaks "mass 4" | The label shows it; it is an OTHER_move over-count. Report its rate; not designed for |
| 15M under-reads a late benefit | Named in §7.2; it falls on adoption, not safety |
| σ_run is uncertain (1–2 df) | Conservative max(prior, internal) σ plus the stage-2 extension |

### 8.2 Open gaps and findings (each a FINDING per standing rule 7)

| id | gap / finding |
|---|---|
| F-X5-1 | **Stale docs.** `ARCHITECTURE.md` §2.1 says 36 tokens and §2.3 says 29; production is 61 (32 event seats). It also says "132" while the code's `MODEL_CONFIG_VERSION` is 134. `delivery_graph.py:973` counts 29. `design_q_head.md` §1 says "29 → 31" (pointer added here). `HiddenOppBeliefPool` reaches the critic too. Out of scope to fix in this doc unit (standing rule 9); reported. |
| F-X5-2 | The blob's physics harm (Jensen gap, attacker gate) is UNMEASURED. X5's justification rests on intent, calibration and Q / search, not on a measured physics loss. |
| F-X5-3 | `rust_env/src/labels/belief.rs:79-81` silently drops a species with no dex number, which would undercount OTHER labels. Make it throw (U1). |
| F-X5-4 | The K9 harness builds a different INIT at 8 threads; `build_learner` must pin threads itself (§6). |
| F-X5-5 | Two different untaught opponents carry the 3.69 and 4.90 floors; they are not one scale. |
| F-X5-6 | The sizing and battery non-inferiority rules used a CI that omits run variance (§7.1). |
| F-X5-7 | The anchors SOP's run floors at 100 games are mostly meter noise (SE ≈ 7 pp per cell). |
| F-X5-8 | `alpha_mask_rate` mixes non-choices with misses. No banked metric isolates the belief-miss share; U4 adds `opp_intent/other_label_rate`. |
| F-X5-9 | `UNDERSTANDING.md` l.586 and l.1346 still call the belief win-rate effect unmeasured; L21381 measured it. Fix in the next UNDERSTANDING pass. |
| F-X5-10 | The KL stop's logging, dose and `update_fit` couplings (§5.3). |
| G-1 | σ_run at 15M, the H2H run floor and snapshot jitter are unmeasured (P0 measures two of the three). |
| G-2 | Negative evidence (an opponent NOT switching to X) is not modelled; X12. |
| G-3 | The hypothesis budget (6 vs 7–8) and OTHER's embedding: `design_q_head.md` §10, read in U8 / the A/B. |
| G-4 | That the Lane S bank exposes both full teams to a reader: UNVERIFIED (U7 checks first). |
| G-5 | PokaiTrainer (arXiv:2608.29197): its belief representation is unread. Read it before X4 returns. |
| G-6 | Whether the op's per-hypothesis attacker rows fit the elementwise budget: unmeasured (U8). |

### 8.3 Build plan (after the deletion pass, before X26)

Sizes are in agent-days. A "tier" is the gate a unit must pass before it lands.

| unit | what | size | tier / gates | agent |
|---|---|---|---|---|
| U0 | P0: checkpoint-vs-checkpoint mirrored H2H CLI (`sprt.py` + Rust eval core) + the P0 reads (§7.3) | 0.5–1 | targeted + static; CPU under `mem_cap.sh` | opus-high |
| U1 | Dex-row table generator (Rust encoder) + committed artifact + `sim`-tier byte-equality gate; `belief.rs` throwing guard | 1 | `sim` + cargo + static | opus-high |
| U2 | T0 hypothesis builder: δ_θ, πps (Tillé), top-k, OTHER mass, masks, set BCE, BeliefHead re-target; the `--belief-tokens` flag, versioning, registry | 2 | routine gate; tier contract; flag gates; πps reference tests | opus-high / xhigh |
| U3 | Tokens into the chain: log-w bias in the transformer and every pool, the op with hypothesis defenders and attackers, OTHER embeddings, I1 / I2 tests per reduction | 2–2.5 | routine gate; invariance tests; obs golden untouched | opus-xhigh (GIGO risk) |
| U4 | Flat α pointer + OTHER labels, re-expressed cells, B ride-along re-base, `other_label_rate` | 1.5 | routine gate; `ridealong_update_test` bit-identity | opus-high |
| U5 | KL early stop per Q1: wiring, logging, controller statistic, realized-step dose, `update_fit`, stop test | 0.5 | routine gate; byte-neutral golden re-record (own commit) | opus-medium |
| U6 | K9 golden: thread pin, `init_group_sha256`, second entry, references, teeth test | 0.5–1 | routine gate | opus-high |
| U7 | Readers: `main.belief_roles` (R1–R4) + the intent / presence / OTHER purpose reads on the Lane S bank | 1 | targeted + static | opus-high |
| U8 | Smoke + the first-two-minutes real launch + GPU cost budget (§3.6) via `gpu_lock` | 0.5 | `--debug` smoke, real launch, learner benchmark | opus-high |
| — | Full suite before ship (`pytest src/ -q`), slow tier on the X5 commit | — | before `/gen3ai-ship` of U2–U6 | — |
| A/B | 6 (→ 10) runs via the training agent | GPU 15.3 (→ 25.5) h | the registration (§7.4) committed before the first game | training agent |

**Total about 9.5–11 agent-days + 15–26 GPU-h.** The backlog's "≈ 2 more attention tokens and a small model-version
bump" understated the build. The token count is right; the physics, pointer and reductions are the work.

**Order.**
- After the deletion pass; U0 can run now (CPU).
- U1 → U2 → U3 → U4 is the chain. U5 and U6 can run in parallel with U2–U4. U7 runs in parallel; U8 last.
- No Rust core runtime change, so M5's parity gates are a re-run, not a rewrite.
- The registration of §7.4, with P0's numbers filled in, is committed BEFORE any A/B game.

---

## 9. Questions for the owner (real choices)

1. **The KL stop:**
   - (a) an active trust region at 1.5 × 0.01, firing on 41–74 % of updates and cutting dose about 35–45 %;
   - (b) a rare safety guard, ≤ 5 % of updates, checked on the group mean;
   - or (a) as its own lever after X26?

   Recommendation (b) + (c): an unmeasured dose-sized change should not ride on the X5 verdict.
2. **The margin you will accept against its price.**
   - About 3.5 pp is buyable at about 15 GPU-h ONLY if P0 shows the head-to-head's run floor is ≤ 1.5 pp.
   - On the untaught meter, the choices are 7 pp (15 GPU-h), 5.5 pp (26 GPU-h) or 4.3 pp (41 GPU-h).
   - Is a 1-in-20 false "non-inferior" (α = 0.05) acceptable for an architecture choice, or do you want the clinical
     1-in-40?
3. **Head-to-head as primary.** It is non-transitivity-prone, but it is the direct question, "is X5 at least as strong
   as blob". Or keep the untaught meter primary and accept the wider margin?
4. **X26 by continuation.** If the A/B arms carry the X26 ride-along heads from step 0 (they are bit-identical to
   learning), the winning arm's seed-1 run can CONTINUE as the X26 baseline, saving a fresh 15M. Or must X26 be its own
   fresh launch?
5. **A learned scale on the log-presence bias** (Graphormer-style), or the plain ToMe identity? Recommendation: plain,
   because it is exactly "w copies" and pinned by I2. A learned scale would make I2 false.
6. **The flat opponent pointer bundled with the tokens** (one retrain boundary, as the north star says), or tokens
   first and the pointer as a second arm? Recommendation: bundle. The pointer needs the hypotheses, and α is detached.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-03 | Presence weights (proposal) | Iterative capped πps (Tillé 2006): Σπ = k exactly | `min(1, k·q)` (`design_q_head.md` §1): loses mass to OTHER whenever a weight caps | §3.2, §3.9 |
| 2026-10-03 | Hypothesis content (proposal) | A dex-row table from THE observation encoder, committed beside the code, byte-gated | A torch-side row builder (a second encoder) | §3.4 |
| 2026-10-03 | Presence in attention (proposal) | Log-w key bias in every reduction (ToMe proportional attention), pinned by I1 / I2 | A learned scale (breaks I2); unweighted reductions | §3.5; Bolya et al. 2023 |
| 2026-10-03 | OTHER (proposal) | Entity with a learned + tail-mean embedding; NO physics rows; masked structurally | Physics at the tail mean (the blob again) | §3.3 |
| 2026-10-03 | Token count (proposal) | 61 → 62: OTHER_species new; OTHER_move re-uses the active's E5 seat | +2 seats | §3.1 |
| 2026-10-03 | A/B statistics (draft for the orchestrator) | Run-level SE (σ_run in the SE), P0 picks the primary meter, K = 3 → 5 two-look O'Brien–Fleming, matched-wall-clock rule | A meter-only CI (F-X5-6); a GSPRT on one seed pair as the population inference | §7 |
| 2026-10-03 | KL stop (draft for the owner) | Recommend a safety guard on the group mean for the A/B; the active trust region as its own lever | Bundled as an active stop at 0.015 per micro-batch (the 2026-10-02 bundle) | §5; Q1 |
