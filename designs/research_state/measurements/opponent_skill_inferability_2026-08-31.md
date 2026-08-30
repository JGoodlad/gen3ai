# SI-1 — opponent skill/class inferability from observable gameplay (measured 2026-08-30)

**First gate of the opponent-skill-conditioning design candidate** (ledger: "🎚️ DESIGN
CANDIDATE: opponent-SKILL conditioning", `8c1c2e8`). Question: from an in-progress battle's
observable history alone — no hidden info, no our-side internals — can a probe predict (a)
the opponent's CLASS (scripted bot vs pool-snapshot "model", and which bot) and (b) the
opponent's anchored ELO, and how fast in turns observed? Probe script:
`opponent_skill_inferability_probe.py` (beside this file); raw numbers:
`opponent_skill_inferability_2026-08-31.json`.

## Verdict in one paragraph

**Bot-vs-model is glaring and fast: 86.9% at ONE observed turn, 90.4% at two, 91.3%
(AUC 0.956) at three, saturating at 94.2%** — and it generalizes unchanged across runs.
One feature carries most of it: **voluntary switch rate** (scripted bots essentially never
voluntarily switch; the trained models switch on ~23% of decisions). **Snapshot ELO among
kin policies is weak and gets WEAKER under honest cross-run evaluation: R² ≈ 0.07–0.10,
Spearman ≈ 0.23–0.27** — a behavioural fingerprint exists but it identifies *the policy*,
not *the skill level*. Both registered predictions scored CORRECT. For the design candidate
this means: a cheap in-context latent can trivially learn "is this a scripted bot, and
roughly which family" within 2–3 turns, but "how good is this model-class opponent" is
mostly NOT recoverable from short observable histories with hand features — the (b)
in-context skill latent should be framed as a CLASS posterior (its supervised label is
free and separable), not as a rating regressor.

## Data

- **20,079 battles** from 13 ai_v9-era runs (gen11–gen17 mainline + the tdaux probe forks:
  `ai_v9_13` … `ai_v9_21`), every eval step's `eval_traces/step_*/<opponent>/*_summary.json`.
  READ-ONLY; no server; no model loads (fully model-free — works on arch-drifted runs).
- Classes: 9 scripted bots (13,595 battles) + "model" = `sentinel_*` pool snapshots (6,484).
- Labels: class = trace directory name. Bot ELO = `data/gen3_bot_elo_anchors.json`.
  Sentinel ELO = the run's `eval_results.jsonl` row at that eval step (ordered `sentinels`
  list = `sentinel_<i>`) → snapshot step → `snapshot_ladder/ladder.json` anchored rating.
  6,464 sentinel battles resolved an ELO; label range 1574–2069, sd 125.
- Model: `HistGradientBoosting` (default params), 5-fold GroupKFold by (run, eval-step) —
  no eval cycle straddles train/test. A simple model is the point: this measures the DATA's
  separability, not a classifier ceiling.

### Feature set (opponent-observable ONLY — the leakage design)

Per observed turn we use only what our side legally sees: the opponent's realized action
(`outcome.opp.action`), their public active HP and revealed-bench count, and the public HP
deltas of the exchange. Move identity is reduced to public move PROPERTIES (basePower,
accuracy, priority, status-ness via `gen3_moves.json`) plus diversity/repetition rates.
17 prefix features: `vol_switch_rate`, `n_species_revealed`, `status_move_rate`,
`heal_rate`, `repeat_rate`, `distinct_move_frac`, `mean/max_dmg_to_us`,
`zero_dmg_attack_rate`, `stay_low_hp_rate`, `switch_low_hp_rate`, `tank_stay_rate`,
`priority_move_rate`, `mean_hp_at_vol_switch`, `faint_repl_per_turn`, `mean_move_bp`,
`mean_move_acc`.

**Excluded by design** (each is a leak of something other than opponent behaviour):
move/species IDENTITY one-hots (would classify the team draw, not the play); our policy
probs / chosen action / recorded belief / α–β intent (our model's internals); reward/value
channels; battle RESULT and total length (a "short battle we won" feature rides the
trainee's 99% win rate vs random — a label shortcut). Residual risks documented below.

## 1. The inferability curve (headline artifact)

| turns observed | bot-vs-model acc | AUC | 10-way balanced acc | sentinel-ELO R² | Spearman | ELO MAE |
|---|---|---|---|---|---|---|
| t1 | 0.869 ± 0.006 | 0.887 | 0.183 | 0.032 | 0.082 | 98 |
| t2 | 0.904 ± 0.005 | 0.940 | 0.271 | 0.146 | 0.292 | 90 |
| t3 | 0.913 ± 0.004 | 0.956 | 0.302 | 0.208 | 0.359 | 87 |
| t4 | 0.921 ± 0.002 | 0.965 | 0.347 | 0.205 | 0.340 | 87 |
| t5 | 0.926 ± 0.004 | 0.969 | 0.359 | 0.166 | 0.301 | 90 |
| t7 | 0.934 ± 0.003 | 0.974 | 0.382 | 0.162 | 0.308 | 90 |
| t10 | 0.935 ± 0.005 | 0.977 | 0.389 | 0.143 | 0.293 | 92 |
| t15 | 0.942 ± 0.004 | 0.982 | 0.414 | 0.143 | 0.298 | 92 |
| t20 | 0.942 ± 0.002 | 0.983 | 0.421 | 0.130 | 0.289 | 93 |
| full | 0.942 ± 0.003 | 0.983 | 0.427 | 0.132 | 0.288 | 92 |

Chance: binary majority class 0.677 (bot); 10-way balanced 0.100. "How many turns until it
knows": **bot-vs-model crosses 90% at t=2**; the marginal turn stops paying after ~t5.

**The ELO curve is NON-monotonic — it peaks at t3–t4 (R² 0.21) and DECAYS with more
observation (0.13 at full game).** The discriminative signal is concentrated in opening
behaviour (early switch/move choices differ by checkpoint age); full-game rates dilute it
with endgame forced play that every skill level executes alike. More observation is not
more skill signal on these features.

### Cross-run robustness (strict: GroupKFold by RUN, 13 groups)

| turns | bot-vs-model acc | AUC | sentinel-ELO R² | Spearman | MAE |
|---|---|---|---|---|---|
| t3 | 0.906 | 0.952 | 0.096 | 0.270 | 93 |
| t10 | 0.933 | 0.976 | 0.041 | 0.187 | 98 |
| full | 0.938 | 0.982 | 0.071 | 0.230 | 96 |

Binary discrimination transfers to unseen runs essentially unchanged. **Sentinel-ELO R²
HALVES** (0.21 → 0.10 at t3): under (run,step) grouping the same snapshot policy appears as
a sentinel in many cycles of its run, so part of the within-run "skill" R² was the
regressor memorizing a specific policy's behavioural fingerprint and looking up its
rating. The honest cross-run skill signal is Spearman ≈ 0.23–0.27, R² < 0.1.

## 2. Per-class confusion

10-way recall at **t3**: model 0.90, staller 0.41, staller_v2 0.32, aggressive 0.25,
setup_sweep 0.22, setup_sweep_v2 0.22, heuristic 0.21, aggressive_v2 0.20, heuristic2 0.16,
**random 0.12** (655/817 predicted "model"). At **t10**: model 0.93, random 0.56 (still
43% → model), staller_v2 0.51, staller 0.44, aggressive 0.43, the max-damage cluster
(heuristic/heuristic2/aggressive_v2/setup_sweep×2) 0.15–0.25. Full matrices are in the
JSON and below.

### 10-way confusion at t3 (rows = truth)

| truth \ pred | aggr | aggr_v2 | heur | heur2 | model | random | setup | setup_v2 | stall | stall_v2 | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **aggressive** | 372 | 211 | 130 | 129 | 24 | 2 | 214 | 234 | 112 | 90 | 0.25 |
| **aggressive_v2** | 223 | 342 | 152 | 148 | 81 | 4 | 270 | 301 | 104 | 91 | 0.20 |
| **heuristic** | 153 | 197 | 328 | 141 | 104 | 3 | 233 | 196 | 165 | 53 | 0.21 |
| **heuristic2** | 169 | 221 | 152 | 283 | 114 | 4 | 240 | 268 | 152 | 122 | 0.16 |
| **model** | 48 | 60 | 74 | 58 | 5850 | 85 | 52 | 60 | 88 | 109 | 0.90 |
| **random** | 1 | 4 | 9 | 8 | 655 | 96 | 5 | 4 | 12 | 23 | 0.12 |
| **setup_sweep** | 133 | 188 | 176 | 162 | 77 | 0 | 358 | 332 | 125 | 60 | 0.22 |
| **setup_sweep_v2** | 196 | 232 | 147 | 153 | 69 | 2 | 323 | 368 | 102 | 61 | 0.22 |
| **staller** | 80 | 118 | 120 | 103 | 38 | 1 | 132 | 136 | 610 | 142 | 0.41 |
| **staller_v2** | 112 | 122 | 75 | 136 | 132 | 5 | 105 | 120 | 208 | 487 | 0.32 |

### 10-way confusion at t10 (rows = truth)

| truth \ pred | aggr | aggr_v2 | heur | heur2 | model | random | setup | setup_v2 | stall | stall_v2 | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **aggressive** | 654 | 153 | 126 | 73 | 2 | 0 | 144 | 217 | 119 | 30 | 0.43 |
| **aggressive_v2** | 280 | 365 | 198 | 122 | 123 | 0 | 176 | 263 | 142 | 47 | 0.21 |
| **heuristic** | 242 | 176 | 355 | 69 | 97 | 1 | 212 | 209 | 170 | 42 | 0.23 |
| **heuristic2** | 190 | 180 | 135 | 257 | 144 | 2 | 154 | 247 | 146 | 270 | 0.15 |
| **model** | 24 | 59 | 32 | 45 | 6020 | 128 | 26 | 24 | 41 | 85 | 0.93 |
| **random** | 0 | 0 | 0 | 0 | 351 | 457 | 0 | 0 | 1 | 8 | 0.56 |
| **setup_sweep** | 260 | 176 | 220 | 82 | 112 | 1 | 285 | 290 | 162 | 23 | 0.18 |
| **setup_sweep_v2** | 265 | 198 | 174 | 128 | 59 | 1 | 203 | 413 | 175 | 37 | 0.25 |
| **staller** | 181 | 94 | 143 | 46 | 10 | 0 | 164 | 159 | 652 | 31 | 0.44 |
| **staller_v2** | 103 | 71 | 60 | 158 | 84 | 5 | 45 | 78 | 125 | 773 | 0.51 |

## 3. Feature attribution — what the "horrible play" detector actually reads

Permutation importance (held-out fold), binary bot-vs-model at t3:
**`vol_switch_rate` 0.275** — an order of magnitude above everything else — then
`mean_hp_at_vol_switch` 0.023, `mean_move_bp` 0.019, `n_species_revealed` 0.012,
`repeat_rate` 0.011, `faint_repl_per_turn` 0.009. 10-way full-game: `vol_switch_rate`
0.198, `mean_hp_at_vol_switch` 0.081, `repeat_rate` 0.040, `status_move_rate` 0.035,
`faint_repl_per_turn` 0.031, `tank_stay_rate` 0.027.

Class means (full game) that make it readable:

| class | vol_switch_rate | zero_dmg_attack_rate | status_move_rate | repeat_rate | stay_low_hp_rate | mean_dmg_to_us | mean_move_acc |
|---|---|---|---|---|---|---|---|
| aggressive | 0.000 | 0.178 | 0.330 | 0.458 | 0.906 | 0.291 | 0.980 |
| aggressive_v2 | 0.028 | 0.168 | 0.312 | 0.412 | 0.891 | 0.302 | 0.980 |
| heuristic | 0.021 | 0.187 | 0.327 | 0.435 | 0.896 | 0.283 | 0.983 |
| heuristic2 | 0.051 | 0.171 | 0.353 | 0.393 | 0.790 | 0.306 | 0.977 |
| **model** | **0.229** | 0.176 | 0.448 | **0.213** | 0.761 | 0.283 | 0.979 |
| **random** | **0.307** | 0.208 | **0.615** | **0.141** | 0.738 | 0.233 | 0.980 |
| setup_sweep | 0.024 | 0.163 | 0.330 | 0.421 | 0.915 | 0.301 | 0.979 |
| setup_sweep_v2 | 0.014 | 0.160 | 0.330 | 0.413 | 0.904 | 0.311 | 0.976 |
| staller | 0.000 | 0.187 | 0.420 | 0.432 | 0.924 | 0.275 | 0.974 |
| staller_v2 | 0.105 | 0.176 | 0.413 | 0.377 | 0.577 | 0.283 | 0.975 |

The owner's "horrible play" intuition scores a partial hit: the loudest tell is not a
single blunder but a POLICY-SHAPE statistic — **whether the opponent ever voluntarily
switches, at what HP it switches, and whether it re-clicks the same move**. The
blunder-shaped features we built for the purpose (`zero_dmg_attack_rate` — attacking into
immunities; `stay_low_hp_rate`) carry almost nothing (importance ≤ 0.006): the scripted
bots are max-damage cores that rarely click immune moves, so "obvious mistakes" are RARER
in bots than the intuition expects — it is rigidity, not stupidity, that identifies them.

## 4. Honest negatives + leakage residuals

1. **The max-damage bot cluster does not separate.** heuristic / heuristic2 /
   aggressive_v2 / setup_sweep / setup_sweep_v2 confuse into each other at 0.15–0.25
   recall even at t10 — they share the greedy-damage core and differ in narrow triggers
   (setup openings, boost thresholds) that category-rate features cannot see. Move/species
   identity features would separate them but were excluded as team-draw leaks.
2. **Random reads as "model", not as a bot.** At t3 random's recall is 0.12 with 80%
   predicted "model": on switch-willingness and status-usage — the dominant features —
   random-legal play and trained-model play share a surface. It takes ~10 turns of
   move-quality accumulation (`mean_dmg_to_us` 0.233 vs 0.283, repetition 0.14 vs 0.21)
   to pull them apart (0.56 recall). A skill scalar that treats "model-like" as "strong"
   would rate a random-mover as mid-ladder for the first several turns.
3. **Kin snapshots do not rank.** The registered-weak leg came in weaker than registered
   once evaluated strictly: cross-run sentinel-ELO R² ≈ 0.07–0.10. Within-run R² 0.21 is
   partly policy-identity memorization (same snapshot recurs as sentinel across cycles).
   ~440 ELO of real ladder spread among sentinels is nearly invisible to behavioural
   category rates.
4. **Residual leak risks accepted + documented:** `mean_dmg_to_us` is a joint quantity
   (their move choice × OUR team's bulk) — legitimate observation but partly trainee-
   dependent; trace SAVING is outcome-quota'd (`win_*`/`loss_*` file caps), so the saved
   battles are not an unbiased sample of eval games (features exclude outcome, but
   behaviour conditioned on saved-battle selection can differ mildly); forced switch-ins
   after our own faints are folded into per-turn records as `faint_repl_per_turn` rather
   than excluded. None of these can manufacture the headline signal (`vol_switch_rate` is
   computed from the opponent's own voluntary actions only).

## 5. Registered predictions — scored

| prediction (pre-registered, not tuned toward) | outcome |
|---|---|
| Bot-vs-trained-model discrimination >90% within ≤3 observed turns | **CORRECT** — 90.4% at t2, 91.3% (AUC 0.956) at t3; holds cross-run (90.6%) |
| Snapshot ELO regression WEAK (kin policies; R² < 0.3 at full game) | **CORRECT, and understated** — R² 0.132 at full game within-run grouping; 0.071 under strict cross-run grouping |

## 6. What this buys the design candidate

- **Menu item (b) — the in-context skill LATENT — is half-viable:** a CLASS posterior
  (bot-family vs model, K-way) is learnable from 2–3 turns of observable history with
  labels the eval pipeline already provides for free, and it generalizes across runs. The
  RATING half is not there: on kin (model-class) opponents, short-history hand features
  recover < 10% of ELO variance. If skill conditioning is built, condition on CLASS (or
  the bot/model dichotomy + coarse family), not on a regressed rating.
- **Menu item (a) — the train-time skill SCALAR — is unaffected** (its labels are exact at
  train time); this probe bounds only what the DYNAMIC posterior could recover at play
  time from behaviour alone with simple features. A learned trunk sees strictly more than
  these 17 rates (exact move choices in board context); treat R² ≈ 0.1 as a floor, not a
  ceiling — but the non-monotonic curve (signal concentrated at t≤4) and the kin-collapse
  are properties of the data the trunk inherits.
- Next gate per the ledger remains the ORACLE A/B (true class as input; if the oracle
  moves nothing, the menu is dead regardless of inferability) and the de-mixing read of
  the existing stratified α metrics.
