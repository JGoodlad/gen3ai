# Is MOTIVATED COGNITION biting the shaping-mode belief heads? (2026-10-07)

CPU-only read of the 16 X5 A/B finals on the Lane S bank. No training, no GPU.

## 0. PRE-REGISTRATION (written 2026-10-07, BEFORE any result was computed)

This section was committed to this file before `extract.py` ran on the full bank. Only a
3-battle feasibility probe (`probe_once.py`) ran first. It printed tensor SHAPES and no belief or
calibration number. Nothing in this section was edited after results existed. §2 onward is written
after the read.

### 0.1 The question

Production trains the belief heads with `belief_grad_mode: shaping`, so all four gradient routes
are open (`src/agents/model/CLAUDE.md`, `--belief-grad-mode`). Route C (PPO loss into the head's own
parameters) lets the policy gradient bend a belief toward whatever makes the preferred action look
right. That is "motivated cognition". Route C is not assumed to be bad: gen-12 convicted
`label_only` for the win-prob head, and the owner recalls the belief heads also did better with PPO
flowing through them. Backlog X20 holds the three-arm experiment (`label_only` / `shaping` / split)
that a positive finding here would justify.

### 0.2 Data and the code pin

- **Checkpoints.** The 16 X5 A/B finals (`final_model.zip`, 15.05M steps), the same seeds as look 3
  (`../x5ab_look3_2026-10-07/README.md`):

  | seed | fixed_mass | blob |
  |---|---|---|
  | 1001–1005 | `rb_x5ab_fm_s100N` | `rb_x5ab_blob_s100N` |
  | 1006 | `rb_x5ab_fm_s1006b` | `rb_x5ab_blob_s1006` |
  | 1007, 1008 | `rb_x5ab_fm_s100N` | `rb_x5ab_blob_s100N` |

  The oracle arms and `rb_x5ab_fm_s1006` are not seeds. All 16 runs record `belief_grad_mode: shaping`.
- **States.** The Lane S bank (`m5_laneS/bank_v1`): 20,712 decisions in 580 battles, re-encoded by
  the pin's own Rust core. Both full teams sit on every battle record, so every label is exact
  (see 0.3 for the item exception).
- **The pin.** Everything executes from the detached checkout at `706fa536`
  (`/home/goodlad/dev/gen3ai/.claude/worktrees/x5-look3-pin-706fa536`). Its `src/` and `data/`
  were verified byte-identical to `git archive 706fa536` on 2026-10-07. cwd and `PYTHONPATH` are
  both the pin (`run_pin.sh`), because `data/` is read from the cwd, and HEAD's
  `gen3_item_priors.json` and `gen3_ability_priors.json` differ from the pin's. The looks-1–2 runs
  were trained at `e5e660dd` / `708dcb0a`, and their learner goldens are identical to `706fa536`
  (look-3 README), so one pin serves all 16.
- **Forward.** The checkpoint goes through the strict loader on the CPU in eval mode, inside
  `inference_globals(4)`, in batches of 256 in bank order. The extra obs keys are zero-filled
  (`zero_extra_obs`), exactly as `main.belief_roles` does.

### 0.3 Part 1: calibration, per head, arm, seed

Every head has three columns: the ARM's posterior, the Smogon PRIOR it fuses with (the head's own
non-persistent prior buffer, so the identical prior for every checkpoint), and ARM − PRIOR.

| head | scored units | label | prior column |
|---|---|---|---|
| **species presence** | (row, s ∈ V): V = the structural candidate set (valid dex-row nums minus the revealed ones; `species_candidates`), rows with k ≥ 1 hidden | 1[s is on the opponent's true team and not revealed] | `fixed_mass_presence(T0 team prior)`. The arm's π is the reader's: fixed_mass = `hs.species.pi`; blob = `belief_head_team_scores` → the same construction |
| **moves** | (row, revealed opp slot j, move m): the m with Smogon prior P(m \| species) ≥ 0.01 (label-independent), NOT already revealed for that slot; Hidden Power is scored on its 16 TYPED channels (355–370), each with prior P(HP) · P(type), and is skipped for a slot whose HP is already revealed (H5 covers that). A slot with 4 revealed moves is skipped | 1[m is in the true set] (typed HP: 1[the set runs `hiddenpower<type>`]) | `MoveBelief.move_prior_probs[species]` |
| **item** | (row, revealed opp slot j) whose item is NOT revealed in the obs (`item_ids == 0`) | the mon's STARTING item num from the packed team (none → 0) | `ItemBelief.item_prior[species]` |
| **nature** (the spread head's categorical) | (row, revealed opp slot j) with a nature in the packed team | nature num | `softmax(SpreadBelief.nature_logprior[species])` |
| **EV** (the spread head's regression, descriptive only) | same slots | packed EVs | `SpreadBelief.ev_prior[species]` |
| **HP type** | (row, revealed opp slot j) whose true set runs typed Hidden Power | the type index (HP_TYPE_NAMES order) | `HPTypeBelief.hp_prior[species]` |

Metrics:

- **Binary heads (presence, moves):** ECE over 10 equal-width bins, Brier, log score (−log p of the
  label, p clamped to [1e-12, 1 − 1e-12]), and calibration-in-the-large CITL = mean p − mean y.
  Reliability is tabulated on edges {0, .01, .02, .05, .1, .2, .3, .5, .7, .9, 1}. For presence,
  Σ_V π = k by construction, so CITL is identically (k − #true hidden in V)/|V|. That makes it
  uninformative, and it is reported only as a check.
- **Categorical heads (item, nature, HP type):** multi-class Brier, log score of the true class,
  top-1 ECE (confidence vs accuracy, 10 equal-width bins), and top-1 CITL = mean confidence −
  accuracy (> 0 means over-confident).
- **Item, one-vs-rest for Choice Band** (T3's quantity): ECE, Brier, CITL.

Each metric is a per-seed value. Across seeds I report the mean and a t 95 % CI (df 7) for ARM, PRIOR
and ARM − PRIOR. Part 1 is DESCRIPTIVE. A head is called over- or under-confident only when the
across-seed CI of its top-1 CITL (categorical) or CITL (moves) excludes 0. Otherwise it is called
calibrated in the large.

**Known label limitation:** the item head is TRAINED toward the mon's CURRENT item (the Rust label
reads the battle's truth state), but the bank stores only the starting team. A consumed berry or a
Knocked-Off item mislabels a few later rows. Choice Band is not consumable, so T3 is unaffected
except by Trick and Knock Off.

### 0.4 Part 2: the SELF-SERVING tests (the core)

**The checkpoint's PREFERENCE** is the greedy argmax of its own masked policy on the row (the eval
regime, as in look 3). It is not the action the bank played. **Rule 8:** a row whose top-1 minus
top-2 probability is < 1e-6 is excluded. **Eligible rows:** `kind == free`, with ≥ 1 legal move
action (6–9, or Struggle 10) AND ≥ 1 legal switch action (0–5). Only there is staying vs switching a
choice. STAY means the argmax is a move or Struggle; SWITCH means it is a switch.

**Signed error** is belief minus truth, so negative means the head UNDER-estimates. Every test is
computed for the ARM column and for the PRIOR column on the same rows. The prior is a function of
species alone and cannot be motivated. **Δ_delta = Δ_arm − Δ_prior** is what the learned delta
(routes A + C) adds on top of the selection effects the prior already shows.

- **T1, move threat, STAY vs SWITCH.** Θ(row) is the set of the opponent ACTIVE's unrevealed
  candidate moves (the moves-head unit set above, at the active slot) that are damaging (base power
  > 0) with type multiplier ≥ 2 against our active's types (a typed HP channel uses its type). It
  drops Ground moves when our active's ability is Levitate. A row needs Θ ≠ ∅. Per row,
  e1 = Σ_{m∈Θ} (p_m − y_m), the error in the EXPECTED NUMBER of hidden super-effective moves.
  **Δ1 = mean e1 | STAY − mean e1 | SWITCH.** Motivated cognition predicts Δ1 < 0: when the policy
  stays in, the threat to the active is under-estimated (or over-estimated when it leaves).
- **T2, hidden counters to the mon the policy COMMITS to the field.** The committed mon Z* is our
  active if the argmax stays, else the switch target. The ALTERNATIVES A are every other mon the
  policy could have had on the field this decision (the legal switch targets, plus our active when
  the argmax switches). A row needs k ≥ 1 and A ≠ ∅. C(Z) = {s ∈ V : some type of s has multiplier
  ≥ 2 against Z's types} (species-level STAB threat). e(Z) = Σ_{s∈C(Z)} (π_s − y_s).
  **Δ2 = mean over rows of [e(Z*) − mean_{Z∈A} e(Z)].** This is a WITHIN-decision contrast.
  Motivated cognition predicts Δ2 < 0: the counters to the chosen mon are under-estimated relative to
  the counters to the mons it passed over.
- **T3, Choice Band, STAY vs SWITCH.** Rows where the opponent active's item is not revealed in the
  obs and its species' Smogon P(Choice Band) ≥ 0.02. e3 = P_arm(CB) − 1[starting item = CB].
  **Δ3 = mean e3 | STAY − mean e3 | SWITCH.** Motivated cognition predicts Δ3 < 0: a CB is
  under-estimated when the policy stays in.

**Uncertainty, two sources, both required:**
(i) across seeds: a t 95 % CI on the 8 per-seed values (df 7);
(ii) the bank: a battle-cluster bootstrap. It resamples the 580 battles with replacement
B = 1,000 times (`numpy.random.default_rng(20261007)`), recomputes each seed's Δ on the resampled
rows and averages over seeds, giving a 95 % percentile CI. (i) cannot see bank sampling noise,
because all seeds share the same rows.

**DECISION RULE, per (test, arm), set before any result:**

- **BITING** iff ALL of the following hold:
  - mean Δ_arm < 0 and mean Δ_delta < 0;
  - for BOTH quantities, BOTH CIs ((i) and (ii)) lie strictly below 0, by more than 1e-9;
  - |mean Δ_delta| ≥ a materiality floor M, by more than 1e-9: M = 0.02 expected moves (T1),
    0.02 expected species (T2), 0.01 probability (T3). M is the smallest effect I would spend a
    3-arm training experiment on.
- **ANTI-self-serving** iff the mirror image holds (> 0, the same CIs and floor): the head is
  pessimistic about the chosen action.
- **NOT DETECTED** otherwise. A CI endpoint, or |mean| vs M, within 1e-9 of its boundary reads NOT
  DETECTED (rule 8).

**Overall:** "motivated cognition BITES" iff any (test, arm) is BITING. **X20 mapping:**

- BITING only on T2 (the species belief): X20 is justified, and the arm it suggests is the SPLIT
  (cut route C on the species belief only).
- BITING on T1 or T3 (per-mon heads): X20 is justified, with `label_only` on those heads.
- No test BITING: this read gives X20 no support, and X20 keeps its current backlog standing with
  this null attached.

### 0.5 Part 3: blob vs fixed_mass

Per test, a Welch two-sample t on the per-seed Δ_delta, blob vs fixed_mass, two-sided at 0.05. Per
calibration metric, the same comparison on ARM − PRIOR. Theory predicts blob is more exposed. Its
species belief is a continuous latent with route C open, while fixed_mass reads its species masses
detached where they are consumed. The per-mon move, item and spread heads are shaping in both arms,
so T1 and T3 are not predicted to differ. **"Blob more exposed"** is claimed only if blob's Δ_delta
is more negative with p < 0.05.

### 0.6 Files

| file | role |
|---|---|
| `run_pin.sh` | runs a script at the pin (cwd + PYTHONPATH), refuses otherwise |
| `probe_once.py` | the shape-only feasibility probe |
| `extract.py` | per checkpoint: the forward, per-row self-serving arrays → `rows/<label>.npz`, per-head calibration aggregates → `rows/<label>.calib.json`; resumable (skips a label whose outputs exist) |
| `analyze.py` | the across-seed / bootstrap read and the decision rule → `result.json`, `result.md` |

### 0.7 FOLLOW-UP T4, added 2026-10-07 AFTER the T1–T3 results (orchestrator request); NOT part of the registered verdict above

T4 is X20's own bending detector, "P(opponent stays) must not inflate when we attack", applied to the
opponent-INTENT head for the **fixed_mass arm only**: the 8 fm seeds, at the same pin, bank and
forward as §0.2. Blob is being deleted. This section was written before `alpha_t4.py` ran.

- **The head:** fixed_mass's flat opponent pointer (`flat_intent_logits`, softmax over its live
  columns). P_sw = its mass on the six switch-slot columns plus OTHER_species. P_stay = 1 − P_sw.
- **The label:** the opponent's realised action at this decision (`BankRows.event`: a switch iff
  event ≥ 1000).
- **Rows:** T1's eligible rows (`free`, both a move and a switch legal for us, rule-8 margin
  ≥ 1e-6), with a labelled event and ≥ 1 live opponent switch column.
- **Our action:** STAY (argmax is a move or Struggle) vs SWITCH, exactly as T1. Descriptively, I
  also report ATTACK (the argmax is a move the bank's `cats` calls `attack`) vs SWITCH.
- **The statistic:** e = P_sw − 1[opp switched]. **Δ4 = mean e | STAY − mean e | SWITCH.** Δ4 < 0
  is self-serving: when we stay in, the head expects too few opponent switches, i.e. P(stay)
  inflated.
- **The prior column:** intent has no Smogon prior, so the prior is the BASE RATE, a constant equal
  to the bank's overall opponent switch rate on these rows. A constant cannot select on state, so
  its gap is −Δ(y), and arm − prior = Δ(P_sw), the head's own swing. Both are reported as asked.
  **The verdict is on Δ4_arm**, the head's calibration gap conditional on our action. Subtracting
  the constant only replaces it with Δp, which is not a calibration quantity.
- **The rule (rule 8):** α BENDS iff mean Δ4_arm < 0 AND both its across-seed t95 CI (df 7) and
  its battle-cluster bootstrap CI (B = 1,000, `default_rng(20261007)`) lie below 0 by more than
  1e-9, AND |mean Δ4_arm| − 0.02 > 1e-9. The mirror image reads ANTI-self-serving; anything else
  reads NOT DETECTED.
- **Secondary (reported, not gating):** as D1, the same Δ4 with the stay/switch split taken from
  another fm seed's policy (the mean over the 7 others). If own-policy and cross-policy agree, the
  gap is the head's, not the policy acting on its own noise.

---

## 1. What ran (2026-10-07, after §0 was committed at `742ccc95`)

- `extract_all.sh` (detached, under `mem_cap.sh 24`): `extract.py` read all 16 finals at the pin.
  The bank was re-encoded in 50 s, with obs as recorded on 19,964 of 19,964 checked rows. Each
  checkpoint took 13–20 s on 4 threads, peak 5.2 GB. Then `analyze.py` produced `result.json` and
  `result.md`.
- Eligible rows: 17,728 per seed (17,727 for two fixed_mass seeds). Rule-8 argmax ties excluded:
  0 for blob, 1 row each in fm s1004 and s1005. The policies STAY on 72–82 % of eligible rows.
- Rows per test, per seed: T1 14,500 (≈ 10,900–11,800 STAY / 2,700–3,600 SWITCH), T2 13,308, T3 2,446.
- `rows/` (the registered inputs) and `rows_v2/` (the §3 rerun, a superset) are archived at
  `~/gen3ai_archive/motivated_cognition_2026-10-07/`, with sha256 in `rows_sha256.txt`. Only the
  per-checkpoint `*.calib.json` aggregates are committed. To rerun `analyze.py` / `explore.py`, copy
  the archive's `rows/` and `rows_v2/` back here.
- Files added after §0: `extract_all.sh` (the detached driver; `ROWS=rows_v2` for §3), `explore.py`
  → `explore.md` / `explore.json` (§3), `rows_sha256.txt`, `extract.log` / `extract_v2.log`. After
  §0, `extract.py` gained only the §3 arrays (`s1_*`, `y1`, `p_stay`, `u_*`). The registered arrays
  are unchanged (D0).

## 2. Result, the registered read (`result.md`)

### 2.1 The self-serving tests: **motivated cognition BITES, on ONE head in ONE arm**

| test | arm | Δ_arm | Δ_prior | **Δ_delta** [t95] [boot95] | verdict |
|---|---|---|---|---|---|
| T1 move threat to our active, STAY − SWITCH | **blob** | −0.122 | +0.014 | **−0.136 [−0.159, −0.113] [−0.153, −0.121]** | **BITING** |
| T1 | fixed_mass | −0.010 | +0.004 | −0.014 [−0.029, +0.001] [−0.020, −0.007] | NOT DETECTED (t-CI straddles 0; below M = 0.02) |
| T2 hidden counters to the committed mon vs the alternatives | blob | +0.007 | −0.003 | +0.010 [+0.001, +0.018] [+0.001, +0.018] | NOT DETECTED (anti-direction; the Δ_arm CI straddles 0; below M) |
| T2 | fixed_mass | −0.007 | −0.004 | −0.003 [−0.008, +0.001] [−0.015, +0.008] | NOT DETECTED |
| T3 Choice Band on the opp active, STAY − SWITCH | blob | −0.006 | −0.032 | +0.026 [+0.017, +0.034] [+0.010, +0.040] | NOT DETECTED (the head REMOVES ~80 % of the prior's selection gap) |
| T3 | fixed_mass | −0.024 | −0.040 | +0.015 [+0.004, +0.027] [+0.002, +0.030] | NOT DETECTED (the head removes 39 % of it) |

**What T1-blob means in plain numbers** (row means, averaged over the 8 seeds). Θ is the opponent
active's unrevealed super-effective moves against our active, counted in expected moves:

| | blob head | Smogon prior | truth | fixed_mass head | prior | truth |
|---|---|---|---|---|---|---|
| when the policy STAYS | 0.417 | 0.346 | 0.236 | 0.369 | 0.348 | 0.240 |
| when the policy SWITCHES | **0.772** | 0.565 | 0.468 | 0.593 | 0.558 | 0.454 |
| over-statement, STAY → SWITCH | +0.18 → **+0.30** | +0.11 → +0.10 | | +0.13 → +0.14 | +0.11 → +0.10 | |

The blob move head over-states the opponent's hidden threats everywhere. It over-states them by
two-thirds more on the rows where the policy leaves than on the rows where it stays, so its error
leans the way that justifies the action taken. Neither the prior nor the fixed_mass head shows this.

**Blob vs fixed_mass (§0.5):** on T1, blob − fm Δ_delta = −0.122, Welch t −10.5, p < 0.001, so
**blob is more exposed**. T2 differs in the ANTI direction (+0.013, p 0.012, not "more exposed").
T3 does not differ (p 0.11).

### 2.2 Calibration per head (across-seed mean; CIs in `result.md`)

| head | blob | fixed_mass | Smogon prior | reading |
|---|---|---|---|---|
| species presence (ECE / Brier / log) | 0.0026 / 0.0060 / 0.0211 | **0.0011 / 0.0046 / 0.0151** | 0.0009 / 0.0060 / 0.0223 | fixed_mass beats the prior on Brier and log score (and blob, p < 0.001). Blob ties the prior on Brier, but its ECE is ~3× the prior's |
| moves of REVEALED mons (ECE = CITL / Brier / log) | **+0.043** / 0.0330 / 0.124 | +0.030 / 0.0290 / 0.111 | +0.024 / 0.0292 / 0.110 | all three OVER-predict hidden moves (CITL CI > 0). **Blob is WORSE than its Smogon prior** on every score; fixed_mass ≈ the prior (Brier Δ −0.0001 [−0.0010, +0.0008]) |
| item (top-1 ECE / CITL / Brier / acc) | 0.028 / −0.023 / 0.221 / 0.858 | 0.029 / −0.018 / 0.229 / 0.853 | 0.081 / −0.013 / 0.359 / 0.764 | both far better than the prior; slightly UNDER-confident (CITL CI < 0) |
| Choice Band, one-vs-rest (ECE / CITL) | 0.021 / −0.003 | 0.025 / −0.011 | 0.085 / −0.076 | calibrated in the large (blob's CI covers 0; fm −0.011, just below) |
| nature (top-1 ECE / CITL / acc) | 0.056 / **+0.034** / 0.714 | 0.064 / **+0.022** / 0.720 | 0.154 / +0.146 / 0.466 | far better than the prior; mildly OVER-confident (CITL CI > 0) |
| HP type (top-1 ECE / CITL / acc) | 0.022 / −0.000 / 0.850 | 0.024 / −0.005 / 0.854 | 0.176 / +0.041 / 0.641 | far better than the prior; calibrated (CITL CI covers 0) |
| EV (MAE, EV points) | 31.7 | 32.0 | 46.2 | both far better than the prior |

Species-presence CITL reads 0.0000 to four decimals, as §0.3 predicts by construction.

## 3. POST-HOC diagnostics of T1 (`explore.py` → `explore.md`; NOT part of the registered verdict)

These were written after §2.1. A frozen checkpoint can show T1's signature WITHOUT route C. The
head may be honest but noisy, and the policy READS it, so the rows where the head happened to
under-state a threat are the rows where the policy stays. Call this "acting on its own noise". The
prior control cannot remove it. `rows_v2/` adds per-unit (p, y) and the T1 sums, and reproduces
every registered array byte-for-byte (D0: 16 of 16).

| diagnostic | blob | fixed_mass | reading |
|---|---|---|---|
| **D1** Δ1_delta, split by the head's OWN policy | −0.136 [−0.159, −0.113] | −0.014 | the registered value |
| D1, split by ANOTHER seed's policy of the same arm | **−0.133 [−0.150, −0.117]** | −0.009 | **unchanged**: not the policy acting on its own head's noise |
| D1, split by the OTHER arm's matched-seed policy | **−0.125 [−0.144, −0.107]** | −0.012 | unchanged, even under a policy that never read this head |
| **D2** STAY − SWITCH error at FIXED belief p (unit level), ARM − PRIOR | **−0.013 [−0.015, −0.011]** | +0.001 [−0.002, +0.004] | blob's head is miscalibrated GIVEN its own p, in the self-serving direction |
| **D3** STAY − SWITCH change in Σp (head) vs Σy (truth) | −0.355 vs −0.233 | −0.224 vs −0.214 | blob's head swings ~1.5× as far as the truth along the stay/switch axis; fm's tracks the truth |

**What D1–D3 settle, and what they do not.**

- **Settled:** the "acting on its own noise" account is ruled out. The bias is a property of the
  blob HEAD in the STATES where policies stay or leave. Every policy's split shows it, because all 16
  policies largely agree on when to stay.
- **What it looks like:** an over-dispersed threat estimate along the decision axis. The head
  exaggerates the evidence for whichever action the state favours. That is what route C would
  produce, since a sharper threat read makes the policy's decision easier.
- **Not settled:** a frozen read cannot prove the cause. Two other accounts survive. The move head's
  label loss may be too weak (coef 0.05) to hold calibration against any other pressure. Or blob's
  head sees inputs that fixed_mass's does not. Only a training contrast that cuts route C on the move
  head can convict route C (X20's `label_only` / split arms, extended to the belief heads).
- **The per-mon move head is `shaping` in BOTH arms, yet only blob's bends.** An open route C is
  therefore not sufficient by itself; something arm-specific carries the pressure. One candidate:
  blob's intent seats are built from the move belief, and fixed_mass's flat pointer is not.
  **UNVERIFIED.**

## 4. Verdict, and what it means for X20

- **Registered: motivated cognition BITES on T1 (the move belief about the opponent's active), in
  the BLOB arm only.** T2 (species) and T3 (Choice Band) are NOT DETECTED in both arms. On T3 the
  item head is the opposite of self-serving: it removes most of the prior's selection gap.
- **§0.4's X20 mapping:** T1 bites, so X20 is justified with `label_only` on the per-mon MOVE head.
  Two qualifications apply:
  - **(a) It bites only in blob.** Under the owner's pre-committed adoption rule, look 3 implies
    fixed_mass, and there T1 is NOT DETECTED. If fixed_mass is adopted, this read gives X20's
    belief-head arms a mechanistic motivation, not a production one.
  - **(b) X20 as written targets a different head.** In `EXPERIMENT_BACKLOG.md` X20 concerns the
    opponent-INTENT head α (`opp_intent_grad_mode`), not the belief heads (`belief_grad_mode`).
    Applying it here needs the owner to widen its scope. This read did not measure α (FINDING 3).
- **Calibration:** the item, Choice Band, nature, HP-type and EV heads are far better than the Smogon
  prior in both arms and close to calibrated. They give no sign that PPO-shaped beliefs are a problem.
  The weak head is MOVES, which over-predicts hidden moves in every column. Blob's move head is worse
  than the prior it fuses with, and fixed_mass's ties it. That is consistent with the 2026-09-24
  off-pool read ("does not beat it even on the pool").

## 5. FINDINGS

1. **Blob's revealed-slot move head is WORSE than its own Smogon prior on the pool** (+0.0038 Brier,
   +0.014 nats/unit). Its error leans toward the action taken: it over-states super-effective threats
   by +0.30 expected moves when the policy leaves, and by +0.18 when it stays. The head feeds the
   damage op.
2. **T1's design cannot separate route C from a weak label.** The registered rule reads BITING.
   D1–D3 exclude "acting on its own noise", but they cannot exclude "the label loss is too weak to
   hold calibration". The causal test is a training contrast, not another frozen read.
3. **Scope mismatch.** The brief maps X20 to the belief heads, but the backlog's X20 is the
   opponent-INTENT head α. Its mode, `opp_intent_grad_mode`, is recorded as `detached` in these runs:
   route B cut, route C OPEN. X20's own bending detector ("P(stay) must not inflate when we attack")
   was NOT run: α is outside this brief's closed head list (standing rule 9). It is cheap to add: the
   same pipeline plus α (blob) or the flat pointer's switch mass (fixed_mass), about 10 min of CPU.
4. **Item labels are STARTING items**, because the bank stores no battle-state item. The head is
   trained on the CURRENT item, so consumed berries and Knocked-Off items mislabel a few rows. T3
   (Choice Band) is nearly immune. The item-calibration row carries the error.
5. **The bank's states are not the X5 policies' own.** Older-generation policies and bots recorded
   them, so every X5 checkpoint is read off-policy. The preference is the checkpoint's greedy argmax
   on those states.
6. **T1's Θ drops Levitate only.** Other ability immunities (Flash Fire, Volt / Water Absorb, Thick
   Fat, Wonder Guard) stay in. The same Θ serves the arm and the prior column, so this blurs both and
   cannot create Δ_delta or the blob-only gap.
