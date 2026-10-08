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
