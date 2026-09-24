# Belief-head calibration off the pool: the MEMORISATION read (2026-09-24)

**Verdict.** MEMORISATION DETECTED, and the heads are NET HARMFUL OFF-POOL. On teams from outside
our 719-team pool, the learned belief heads are WORSE than the free Smogon prior they are fused
with. That holds for the species side readout (−1.28 nats on ladder teams) and for the channel that
reaches decisions, the reinjected revealed-slot move posterior (−7.8 pp hidden-move recall vs the
prior). On ladder teams, 15.8% of hidden slots carry a >0.8-confident belief in a species the
opponent does not have. **Tag: MEASUREMENT · memorisation DETECTED (species D_b +1.61 nats
[+1.38, +1.85]) · NET HARMFUL off-pool (species A_ladder −1.28 [−1.47, −1.10]; revealed-move recall
A_ladder −0.078 [−0.089, −0.066]) · CW flag FIRES.**

Pre-registration: [`PREDICTION.md`](PREDICTION.md) (`3650991a`, committed before any number). Every
registered endpoint is reported. One arm was added AFTER the read and is labelled EXPLORATORY (§4).

## 1. What ran

* **Model:** `ai_v13_22_popr1_loop/final_model.zip` (103,219,200 steps, sha256 `88fc961a819abe5f…`),
  loaded by its own pin `6eb9c776` (detached worktree, its own Rust `sim_bridge`, cwd = pin tree
  whose `data/` equals main's) through `load_model_snapshot` + `check_compatible`: **no ArchDrift**.
* **Battles:** in-process Rust bridge, CPU, nice 15, 4 workers, concurrency 1, `torch` 1 thread.
  Trainee = the model GREEDY. Opponent = a second copy of the model GREEDY. Battle `i` shares its
  trainee team and sim seed across arms. **Bit-reproducible:** the same battles run twice gave
  identical rows.
* **n:** 400 per arm attempted, **395 paired battles** kept (5 indices dropped from all arms, §5),
  **57,238 trainee decisions** (pool 18,879 · ladder 20,001 · procedural 18,358).
* **Ladder set:** Metamon `hl_05_26/gen3ou` rev v5, 22,862 teams (fingerprint in
  `out/manifest.json`). Filter: 22,862 are Showdown-legal; **1,775 (7.8%) fail the procedural
  generator's coverage predicates** (1,727 of them on Sleep Talk, which `isModeledMove` excludes);
  **5,933 (26.0%) have a 6-species set identical to a pool team** and are dropped as in-pool.
  **15,154 pass (66.3%).** 400 were drawn by seed, and 0 were rejected by the pinned
  `Gen3Teambuilder`. For information, the pool itself passes the same predicates at 689/719 (Sleep
  Talk ×29, Will-O-Wisp ×1), and is not filtered.
* **Procedural:** `ou_random_teams.js`, coupled, seed 20260924; 0.43% of move-slot prior mass
  renormalised away; 6 in-pool species sets dropped.

## 2. Headline: SPECIES (`BeliefHead`, hidden slots, matched as trained)

NLL in nats. A = prior − head (positive means the head beats the Smogon prior). D_b = A_pool −
A_ladder is the pool advantage that does not transfer. 95% paired bootstrap over battles.

| k revealed | head NLL a / b / c | prior NLL a / b / c | A pool | A ladder | A procedural | D_b | D_c |
|---|---|---|---|---|---|---|---|
| 1 | 2.24 / 4.37 / 4.96 | 3.25 / 3.52 / 3.50 | +1.01 [0.79, 1.26] | −0.85 [−1.33, −0.38] | −1.46 [−1.78, −1.16] | +1.86 [1.33, 2.39] | +2.47 [2.10, 2.88] |
| 2 | 2.51 / 4.22 / 5.55 | 3.19 / 3.38 / 3.60 | +0.68 [0.52, 0.84] | −0.84 [−1.11, −0.60] | −1.95 [−2.21, −1.74] | +1.52 [1.21, 1.84] | +2.63 [2.36, 2.93] |
| 3 | 2.70 / 4.69 / 5.72 | 2.91 / 3.35 / 3.81 | +0.21 [0.02, 0.41] | −1.34 [−1.63, −1.08] | −1.91 [−2.15, −1.69] | +1.54 [1.23, 1.88] | +2.11 [1.81, 2.43] |
| 4 | 2.61 / 4.72 / 6.04 | 2.65 / 3.05 / 3.86 | +0.04 [−0.20, 0.26] NOT DETECTED | −1.67 [−2.01, −1.33] | −2.18 [−2.48, −1.89] | +1.70 [1.28, 2.12] | +2.22 [1.84, 2.58] |
| 5 | 2.54 / 4.34 / 7.13 | 2.38 / 2.64 / 4.61 | −0.17 [−0.49, 0.13] NOT DETECTED | −1.70 [−2.10, −1.34] | −2.52 [−3.03, −2.06] | +1.53 [1.06, 2.03] | +2.35 [1.79, 2.95] |
| **all** | **2.55 / 4.49 / 5.90** | **2.88 / 3.21 / 3.86** | **+0.33 [0.20, 0.46]** | **−1.28 [−1.47, −1.10]** | **−2.03 [−2.22, −1.86]** | **+1.61 [1.38, 1.85]** | **+2.36 [2.16, 2.59]** |

The memorisation gap on raw head NLL is Δ_b = +1.94 [+1.63, +2.25] and Δ_c = +3.35 [+3.07, +3.66].
The non-transferring share D_b / A_pool is 4.9× [3.7, 7.4]: the head loses off-pool about five
times what it gains over the prior on-pool.

**Confidently wrong** (top-1 prob > 0.8 AND that species is not among the opponent's hidden mons;
the prior never reaches 0.8 on (a)/(b), so its rate is 0):

| k | pool | ladder | procedural | ladder − pool |
|---|---|---|---|---|
| 1 | 0.120 [0.095, 0.145] | 0.193 [0.140, 0.247] | 0.196 [0.166, 0.228] | +0.073 [0.016, 0.134] |
| 2 | 0.097 [0.079, 0.116] | 0.150 [0.125, 0.178] | 0.213 [0.188, 0.239] | +0.053 [0.021, 0.087] |
| 3 | 0.105 [0.086, 0.127] | 0.177 [0.152, 0.204] | 0.200 [0.170, 0.231] | +0.071 [0.037, 0.104] |
| 4 | 0.075 [0.056, 0.099] | 0.147 [0.121, 0.174] | 0.162 [0.133, 0.194] | +0.071 [0.037, 0.107] |
| 5 | 0.090 [0.057, 0.131] | 0.133 [0.101, 0.170] | 0.166 [0.129, 0.204] | +0.043 [−0.008, 0.093] NOT DETECTED |
| **all** | **0.095 [0.084, 0.108]** | **0.158 [0.143, 0.176]** | **0.189 [0.172, 0.207]** | **+0.063 [0.042, 0.085]** |

**The registered decision flag FIRES:** the ladder CW lower bound is above 0.05 at k = 3, 4 and 5
(0.152 / 0.121 / 0.101).

**Calibration:** head ECE 0.260 / 0.395 / 0.471, against the prior's 0.017 / 0.032 / 0.111. The head's
mean top-1 confidence is ~0.60 in EVERY arm, while its accuracy falls 0.35 → 0.21 → 0.14. Accuracy
alone barely separates head and prior off-pool (A_ladder on accuracy +0.024 [−0.002, 0.051], NOT
DETECTED): the damage is confidence, not ranking.

## 3. The other heads (all strata; full per-stratum tables in `out/tables.md`)

| Family (metric) | head a / b / c | prior a / b / c | A pool | A ladder | A proc | D_b |
|---|---|---|---|---|---|---|
| **Moves, revealed slots (BCE)**, the reinjected channel | 0.0154 / 0.0286 / 0.0209 | 0.0145 / 0.0210 / 0.0148 | −0.0009 [−0.0014, −0.0005] | −0.0076 [−0.0080, −0.0072] | −0.0061 [−0.0065, −0.0057] | +0.0067 [0.0061, 0.0072] |
| **Moves, revealed (hidden-move recall)** | 0.590 / 0.441 / 0.448 | 0.579 / 0.518 / 0.540 | +0.011 [−0.008, 0.029] NOT DETECTED | −0.078 [−0.089, −0.066] | −0.092 [−0.103, −0.080] | +0.089 [0.067, 0.110] |
| Moves, hidden slots (BCE) | 0.0587 / 0.0608 / 0.0590 | 0.0359 / 0.0422 / 0.0394 (Smogon mixture) | −0.0228 | −0.0186 | −0.0196 | −0.0042 [−0.0048, −0.0036] |
| Moves, hidden slots (recall@4) | 0.105 / 0.095 / 0.086 | 0.284 / 0.239 / 0.203 | −0.180 | −0.144 | −0.117 | −0.036 [−0.056, −0.014] |
| Item (NLL), item not yet shown | 0.400 / 0.871 / 1.404 | 0.839 / 0.612 / 0.971 | +0.439 [0.348, 0.533] | −0.259 [−0.382, −0.155] | −0.433 [−0.586, −0.289] | +0.698 [0.568, 0.851] |
| Item (accuracy) | 0.860 / 0.768 / 0.707 | 0.720 / 0.779 / 0.631 | +0.141 | −0.011 NOT DETECTED | +0.075 | +0.151 [0.107, 0.197] |
| HP type (NLL) ⚠️ ladder contaminated | 0.444 / *11.6* / 1.458 | 0.767 / *8.34* / 0.709 | +0.323 [0.252, 0.393] | *not interpreted* | −0.749 [−0.926, −0.584] | — |
| Spread: nature CE | 0.572 / 2.204 / 2.148 | 1.317 / 0.876 / 0.853 | +0.745 | −1.328 | −1.295 | +2.073 [1.900, 2.240] |
| Spread: EV MAE (EV points) | 22.2 / 38.7 / 40.2 | 42.0 / 36.4 / 37.4 | +19.8 | −2.3 | −2.8 | +22.2 [20.1, 24.2] |
| Spread: derived-stat MAE (stat points) | 8.0 / 15.2 / 15.2 | 16.0 / 13.4 / 13.4 | +8.0 | −1.9 | −1.8 | +9.9 [9.0, 10.8] |

* **Revealed-slot moves (decision path).** The reinjected posterior does NOT beat the plain Smogon
  move prior even on uniform pool teams (BCE A_pool slightly negative; recall NOT DETECTED). Off
  pool it is clearly worse: −7.8 pp recall of the still-hidden moves on ladder teams and −9.2 pp on
  procedural ones. The per-set knowledge it does have is concentrated on the five stable-opponent
  teams (§4).
* **Hidden-slot moves: STRUCTURAL, not memorisation.** The hidden-slot posterior is a
  state-independent constant: its maximum deviation across all 57k decisions is exactly 0.0, as the
  code implies (the `BeliefSlots` constant token plus the flat-floor prior row). It is far worse
  than the Smogon mixture `Σ_s P_T0(s | revealed) · P(m | s)` in EVERY arm (recall@4 0.10 vs 0.28 on
  pool). D is negative: it is uniformly uninformed, not pool-tuned. Every input of that mixture
  already exists in the forward (`T0SpeciesPrior` plus the move prior buffer).
* **HP type on ladder: contaminated as pre-flagged.** 92.3% of the sampled ladder teams' Hidden
  Powers are HP DARK (Metamon's untyped "Hidden Power" with all-31 IVs), against 0.0% Dark in both
  the pool and the procedural sets. The pool's top types are Grass 42% / Bug 17% / Fire 13%.
* **Item, spread:** Metamon fills unrevealed fields from usage stats and (c) samples them from
  Smogon, so both arms favour the prior BY CONSTRUCTION (§6). These rows are bounded, not clean
  transfer evidence. On pool the heads' advantages are large (item NLL +0.44, EV MAE −19.8 points).

## 4. EXPLORATORY (added after the read, NOT pre-registered): the five stable-opponent teams

While interpreting the read, the run's TensorBoard showed **`train/stable_fraction` 0.36**
(self-play 0.54, bots 0.10). Both stable opponents (`ai_v13_18_teach5_offense_hidose`,
`ai_v13_13_exploit5_offense`) pin the SAME five sample offense teams, so about 36% of this run's
belief labels come from five teams. The training-time belief metrics are correspondingly high:
`belief/species_acc` 0.72 and `species_ce` 0.89, against 0.35 and 2.55 on the uniform pool here.
An extra arm "five" (the opponent pilots those 5 teams, cycled `i % 5`; same trainee teams, seeds
and pilot; 400/400 battles OK, 15,869 decisions) gives, head vs prior:

| | species NLL | species acc | species CW | species ECE | revealed-move recall | item acc | HP acc | nature acc |
|---|---|---|---|---|---|---|---|---|
| **five** head | 1.56 | 0.550 | 0.061 | 0.112 | 0.691 | 0.909 | 0.964 | 0.927 |
| five prior | 3.15 | 0.172 | 0 | 0.040 | 0.567 | 0.712 | 0.547 | 0.540 |
| A five | +1.59 [1.49, 1.69] | +0.378 | — | — | +0.124 [0.109, 0.139] | +0.196 | +0.417 | +0.387 |

The head-over-prior advantage on species is therefore a GRADIENT in how close the teams are to the
label distribution: **five +1.59 → uniform pool +0.33 → ladder −1.28 → procedural −2.03 nats.**
This is the memorisation signature, and its target is mostly five teams rather than 719. Record:
`out/exploratory_five/`.

## 5. Hazards and findings (reported, not fixed)

* 🚨 **Heal Bell crashes the observation encode.** `UnknownVolatileError: volatile 'healbell' has no
  gen3 encoding slot` (`agents/observation/gen3_effects.py`, via `active_context.encode`). Gen 3
  Showdown emits `-activate|<mon>|move: Heal Bell`, which poke-env records as a volatile. It
  happened in 4 of 1,200 battles (ladder i=7; procedural i=152, 310, 324). **Main has the same
  gap** (`healbell` is absent from main's `gen3_effects.py`). The pool contains **0** Heal Bell
  teams, so training never met it; 440 of the 22,862 Metamon teams (1.9%) carry Heal Bell. **A live
  ladder game against a Heal Bell user whose Heal Bell fires would crash our player's encode.**
  `ladder_drift_scan` checks protocol keywords, so it would not catch this.
* **The known port lock bug** (`gen3_locked_request_move_v1`, fixed on main after the pin): a
  Rollout-locked mon's request offered "Solar Beam", and poke-env asserted (procedural i=60). The pin
  tree still carries it. It is known and has no pool exposure.
* **Pool vs training distribution.** The registered pool arm is the UNIFORM 719, which is only the
  non-stable 64% of the training opponent mix (§4). "Pool" numbers here are therefore not the
  training-time metrics.
* **The species head is a side readout.** It never enters pi/vf, and the fed-forward species belief
  is the parameter-free Smogon `T0SpeciesPrior`. Its over-confidence reaches decisions only through
  the trunk it shapes (`belief_grad_mode shaping`). The revealed-slot move rows are the
  decision-path evidence.
* **5 dropped indices** (1 ladder, 4 procedural failures; 0 timeouts) were removed from all arms.
  That is 1.25% of indices, far under the 25% INCONCLUSIVE bar. Selection effect: ladder and
  procedural teams whose Heal Bell fires, or which carry a Rollout lock, are under-represented.
* **Sleep Talk sets** are excluded from (b) and (c) by the registered coverage filter but present
  in (a) (29 pool teams).
* **Trainee win rate differs by arm** (pool 0.51, ladder 0.66, procedural 0.72; greedy self vs
  greedy self). Battle length differs slightly (43.6 / 46.9 / 43.0 turns). Stratifying by k controls
  how much is revealed, but not the play.

## 6. UNVERIFIED

* **UNVERIFIED:** how much of Metamon's species content is filled rather than revealed. Metamon
  documents a fill for unrevealed slots from usage stats. If much of it is filled, (b) is partly
  prior-generated on species too, which would make D_b a LOWER bound for true ladder teams.
* **UNVERIFIED:** that greedy-self reveal patterns match stochastic-training reveal patterns. The
  five-arm species acc is 0.55 here against 0.72 in the training TensorBoard, and that residual is
  unexplained. Candidates: the Hungarian-matched vs logged metric, the reveal regime, and the
  training minibatch mix.
* **UNVERIFIED:** any effect on WIN RATE. This read measures belief quality, not play strength
  off-pool.

## 7. Predictions, scored

| # | Prediction | Outcome |
|---|---|---|
| 1 | A_pool ≈ +1 nat, rising with k | **WRONG**: +0.33, FALLING with k (+1.01 at k=1, NOT DETECTED at k ≥ 4) |
| 2 | D_b > 0 at 40–70%; A_ladder > 0, not net harmful | **WRONG on size**: D_b detected at 4.9× A_pool; NET HARMFUL |
| 3 | A_proc ≤ ~0, D_c > D_b | RIGHT (−2.03; 2.36 > 1.61) |
| 4 | CW pool 10–25% at k ≥ 3; ladder +3 to +10 pp; flag fires; ECE worse off-pool | Pool CW 7.5–10.5% (at or below the band); gap and flag RIGHT; ECE RIGHT |
| 5 | Revealed moves: A_pool > 0; D_b > 0 but a smaller share than species | **WRONG**: A_pool ≤ 0 on BCE, NOT DETECTED on recall; D_b > 0 right |
| 6 | Hidden moves constant; the mixture wins on b/c, loses on a | Constant RIGHT (0.0); the mixture wins in ALL arms, so "loses on a" is **WRONG** |
| 7 | Item: small A, D small or not detected | **WRONG**: A_pool +0.44, D_b +0.70 (Metamon-fill caveat) |
| 8 | HP: A_pool > 0; ladder row contaminated | RIGHT (+0.32; 92.3% Dark) |
| 9 | Spread: A_ladder, A_proc ≤ 0 on nature CE (by construction) | RIGHT (−1.33 / −1.30) |

## 8. Files

* `PREDICTION.md`: the registration.
* `scripts/prep_teams.js` (filter + generator), `scripts/prep_manifest.py` (pinned-teambuilder
  packing + paired draws), `scripts/run_arm.py` (battles + scoring), `scripts/analyze.py`
  (bootstrap), `scripts/render.py` (tables), `scripts/prep_five_arm.py` (exploratory arm).
* `out/manifest.json`: every battle's team shas, seeds, filter counts, Metamon fingerprint and model
  provenance. Packed teams and the Metamon download stay outside the repo
  (`/tmp/belief_cal/`, `~/gen3ai_archive/metamon_cache_2026-09-24/`).
* `out/results.json` + `out/tables.md`: every registered endpoint × stratum × arm with CIs.
* `out/exploratory_five/`: the §4 arm.
