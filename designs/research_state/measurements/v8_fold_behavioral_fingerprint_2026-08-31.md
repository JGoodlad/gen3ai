# M4 — the BEHAVIORAL FINGERPRINT of v8's gift to untaught teams

**2026-08-31 · mechanism-map category #4 (content generalizability).** Probe P
(`v8_redistribution_pfsp_2026-08-30.md`) established **THAT** v8's fold radiated to teams it never
taught — **+5.42pp [+3.44, +7.42], z=4.83, 14/16 untaught teams positive** — beside **+26.18pp** on
its taught controls. It said nothing about **WHAT** the fold made the model *do* differently. This
probe is the instrument for that question.

## THE HEADLINE

**The untaught fingerprint is NOT a weaker version of the taught one. It is a different one — on
the load-bearing axes, an OPPOSITE one.** Over 25 behavioural axes measured on identical boards,
cosine(untaught, taught) = **0.14** (permutation null p = 0.19, sign agreement 12/25 = chance,
through-origin R² = 0.020), against a noise ceiling of **0.60** that the two vectors' own
split-half reliabilities (0.72 / 0.50) would permit. Holding archetype fixed makes it *worse*, not
better: untaught-defensive vs taught reads cosine **−0.12**.

The single sharpest line: **on the teams it taught, the fold SWITCHES MORE and ATTACKS LESS
(`attack_at_all` −6.4pp, z=−8.6; `switch_to_resist` +2.4pp on a 7.7% base, z=+3.3). On the teams it
did not teach, it SWITCHES LESS — concentrated in exactly the reactive situations
(`switch|losing_matchup` −3.9pp, `switch|early` −3.8pp, `switch|behind_on_mons` −2.7pp, z=−5.3) —
and its attack rate does not move.** Both changes gain win rate. They are not the same content at
two doses.

**This is direct evidence for branch (ii) of the ledger's `(i)/(ii)` fork** (scorecard verdict,
2026-08-31): the taught-side fold metrics do not measure the thing that produced v8's untaught
externality, because the untaught externality is not made of taught-side content.

**The gen-era arm ran too** (§6b), and it sharpens rather than repeats the point. The rev-3 fold —
which gifted nothing (−0.39pp here, −0.75pp in probe Q) — changes untaught behaviour **1.9× more
than v8 did** by vector norm. Amount is not the difference; direction is. The two untaught vectors
are orthogonal (cosine 0.11, permutation p = 0.50) and split cleanly: **cutting reactive pivots out
of bad matchups is SHARED by both folds and therefore discriminates nothing**, while every axis
about *what to do when you are winning or when precision is on offer* is **OPPOSED** — the
non-gifting fold got greedier (`take_SE_attack` **+5.1pp**, `switch|ahead_on_mons` **−5.5pp**) where
the gifting fold got more selective (**−4.5pp** and **+1.5pp**).

**Artifacts** — the committed JSON is regenerated *from the committed rows*, so every table below
is re-derivable without replaying a single battle:

| file | holds |
|---|---|
| `…_2026-08-31.json` | every number in this document, both families + the cross-family block |
| `…_2026-08-31_rows_v8.jsonl.gz` · `…_rows_gen.jsonl.gz` | **all 94,053 dual-scored decision rows** — the per-action class vector, the board strata, and both arms' argmax. New axis definitions are testable offline; only the two 11-float probability vectors were dropped (≈⅔ of the bytes, read by no axis) |
| `…_2026-08-31_cells.jsonl` | per-cell win counts + per-game outcome vectors (the win-rate provenance) |
| `…_2026-08-31_gen_selection.json` | the gen-arm team selection, with its derivation recorded |
| `v8_fold_behavioral_fingerprint_probe.py` | the dual-scoring battle pass |
| `v8_fold_behavioral_fingerprint_analyze.py` | axis tables, shape, attribution |

---

## 1. Why behaviour, and not parameters

Ledger `d392e80`: v8's trunk-heavy parameter deltas name nothing in the current architecture, so a
parameter- or representation-space trace of the gift does not port across the ai_v9 rewrite.
*Which decisions changed, on which boards* is architecture-free — it ports with zero translation.
That is the property that makes this probe worth running, and it is why the axis definitions below
are built from the **server-authoritative `LegalActions` snapshot plus `gen3_data` move facts**,
never from the obs vector: nothing here depends on the era's observation layout.

## 2. Method — dual scoring on IDENTICAL boards

Probe P's cells are re-played with per-decision logging, and **at every decision the other arm is
scored on the same `(obs, action_mask)`**. One extra forward per decision, no extra battles. Every
row carries *what the acting arm did*, *what the other arm would have done*, and *the board it
happened on*, so an axis effect is a **paired difference over the same rows** — never a comparison
of two different games.

| | |
|---|---|
| arms | `parent` = `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` · `fold` = `ai_v8_14_distill3_0725/final_model_interrupted.zip` |
| fixed reference opponent | `ai_v8_03_zarch_control_0718` — an ancestor of BOTH arms, equal to neither (the parent-as-opponent alternative makes the parent arm a self-mirror) |
| cells | probe P's own pre-registered selection: 16 untaught probe teams + 6 taught controls, × its 8 fixed opponent teams, 4 games each, both arms |
| totals | **1,408 battles · 54,859 dual-scored decisions** (untaught 1,024 battles / 37,217 decisions; taught 384 / 17,642) |
| CRN | the per-cell seed generator is probe P's verbatim (`random.Random(f"{team}:{opp}")`), so these battles are a **prefix subsample of probe P's own battles**, not a fresh draw |
| play | greedy both sides (`stochastic=False`), node bridge, no server, CPU, `nice 15`, ≤2 cores |
| era pin | the v8 arms load only under `b13b30b`; the battle pass ran from an era-pinned worktree |

**Both arms act**, so both state distributions are covered — a one-sided read would confound the
policy change with the state-distribution shift the policy change causes. The pooled delta is over
a mixture; §3.2 splits it.

### 2.1 The instrument checked itself, three ways

- **ACID (parameter space).** `‖parent−fold‖₂ = 238.9`, `‖parent−ref‖₂ = 53.3`, `‖fold−ref‖₂ =
  192.9`; all three arms distinct, 3,512,397 parameters each. (The fold sits *farther* from its own
  parent than from its grandparent — it simply travelled a long way; distinctness is the gate, the
  lineage-order reading is informational.)
- **Era-pinned load, re-verified in this run rather than imported.** The fold's checkpoint
  reproduces the fold's own recorded `step_292000005` traces at centred-logit **r = 0.98225**, V
  **r = 0.99178**; the parent scores **0.84357 / 0.93719** on the same traces. The load
  discriminates correctly and matches probe P's figures to five decimals.
- **The win rate reproduces probe P.** This probe's 4-of-30-games CRN subsample reads
  **untaught +5.66pp** (probe P: +5.42pp) and **taught +22.92pp** (probe P: +26.18pp) — same sign,
  same order, on far fewer games. Cell identity, seeds and arms did not drift.

### 2.2 The axes

25 axes in five groups, each an indicator on the **chosen action index**. Filters never depend on
which arm chose; that is what keeps every difference paired.

- **A · class shares** on a *free* choice (not a forced switch, not trapped, both families
  available): switch / attack / setup / recover / status / protect / phaze / hazard / other-status.
- **B · conditional switching** — by matchup sign, our HP, opponent HP, game phase, boost state,
  mon-count position.
- **C · move quality** — takes the super-effective attack when one exists; takes the
  `base_power × effectiveness` maximiser; attacks at all.
- **D · switch-target quality** — switches *to a mon that resists the opponent's types*, when such
  a target exists.
- **E · the forced replacement** — a different decision problem (who comes in after a KO).

**Matchup sign is a model-free TYPE proxy** (best STAB effectiveness each way). It does not know
either side's moveset; it is a stratifier, not a claim. **There is no lead-choice axis and there
cannot be one** — gen 3 has no team preview, so the lead is team slot 1 by construction.
*Sacrifice timing* enters as `switch|low_hp(<1/3)`, whose complement is staying in to die.

Statistics: cluster bootstrap over **TEAMS**, 4,000 resamples, point estimate = equal-weight mean
over teams — probe P's own convention, so the two are comparable.

---

## 3. AXIS TABLE — UNTAUGHT (16 teams, 1,024 battles, 37,217 decisions)

Δ is *fold − parent on the same boards*. "realized Δ" is each arm's rate on its **own**
trajectories; realized minus paired is the state-distribution contribution.

| axis | n | parent rate | Δ (fold−parent) | 95% CI | z | realized Δ |
|---|---:|---:|---:|---|---:|---:|
| `switch\|behind_on_mons` | 8892 | 0.214 | **−0.0271** | [−0.0361, −0.0162] | −5.29 | −0.0232 |
| `switch\|high_hp(>2/3)` | 19173 | 0.254 | **−0.0247** | [−0.0352, −0.0146] | −4.63 | −0.0241 |
| `other_status_rate` | 30274 | 0.111 | **+0.0157** | [+0.0083, +0.0237] | +4.05 | +0.0111 |
| `take_SE_attack\|SE_available` | 7913 | 0.643 | **−0.0452** | [−0.0688, −0.0217] | −3.78 | −0.0552 |
| `switch\|early(turn<=8)` | 8192 | 0.304 | **−0.0378** | [−0.0607, −0.0148] | −3.22 | −0.0237 |
| `switch\|losing_matchup` | 6984 | 0.366 | **−0.0385** | [−0.0641, −0.0140] | −3.01 | −0.0327 |
| `switch_rate` | 30274 | 0.233 | **−0.0146** | [−0.0247, −0.0048] | −2.89 | −0.0147 |
| `hazard_rate` | 30274 | 0.039 | **−0.0052** | [−0.0090, −0.0018] | −2.87 | −0.0089 |
| `switch\|even_matchup` | 13243 | 0.214 | **−0.0200** | [−0.0370, −0.0035] | −2.29 | −0.0184 |
| `switch\|ahead_on_mons` | 8082 | 0.213 | **+0.0148** | [+0.0010, +0.0279] | +2.17 | +0.0010 |
| `switch_to_resist\|resist_avail` | 15195 | 0.158 | **−0.0154** | [−0.0311, −0.0010] | −2.05 | +0.0020 |
| `phaze_rate` | 30274 | 0.007 | **+0.0031** | [+0.0003, +0.0067] | +1.86 | −0.0025 |
| `switch\|opp_low_hp(<1/3)` | 3301 | 0.130 | +0.0092 | [−0.0037, +0.0215] | +1.40 | +0.0072 |
| `switch\|low_hp(<1/3)` | 4046 | 0.182 | +0.0171 | [−0.0047, +0.0434] | +1.38 | +0.0080 |
| `switch\|we_are_boosted` | 2101 | 0.069 | +0.0097 | [−0.0045, +0.0241] | +1.32 | +0.0331 |
| `forced_repl_resists\|resist_avail` | 2003 | 0.460 | −0.0197 | [−0.0510, +0.0091] | −1.29 | +0.0076 |
| `status_rate` | 30274 | 0.060 | +0.0041 | [−0.0027, +0.0114] | +1.14 | +0.0018 |
| `take_best_damage\|>=2_attacks` | 17198 | 0.397 | −0.0080 | [−0.0234, +0.0070] | −1.01 | +0.0023 |
| `attack_at_all\|>=2_attacks` | 17198 | 0.585 | +0.0065 | [−0.0066, +0.0205] | +0.95 | +0.0273 |
| `switch\|winning_matchup` | 10047 | 0.166 | +0.0071 | [−0.0079, +0.0218] | +0.94 | −0.0029 |
| `recover_rate` | 30274 | 0.029 | −0.0044 | [−0.0150, +0.0055] | −0.85 | −0.0019 |
| `attack_rate` | 30274 | 0.474 | +0.0026 | [−0.0076, +0.0125] | +0.50 | +0.0247 |
| `switch\|late(turn>=20)` | 11430 | 0.158 | −0.0033 | [−0.0165, +0.0097] | −0.49 | −0.0240 |
| `setup_rate` | 30274 | 0.030 | −0.0005 | [−0.0037, +0.0023] | −0.35 | −0.0067 |
| `protect_rate` | 30274 | 0.017 | −0.0006 | [−0.0056, +0.0041] | −0.25 | −0.0029 |

**Divergence:** argmax disagreement **0.303** · mean KL(fold‖parent) 0.242 · **flip mass involving
SWITCH 0.510** · given both arms switch, they pick the same mon **74.8%** of the time (n=9,689) ·
given both move, the same slot 77.2% (n=24,203) · games are **shorter** (37.3 → 35.4 decisions per
side, −5.2%).

**Class flow** (parent count → fold count, and how leaky each class is):

| parent class | parent n | fold n | net | flip-out rate |
|---|---:|---:|---:|---:|
| SWITCH | 11565 | 11120 | **−445** | 0.373 |
| OTHER_STATUS | 3949 | 4318 | **+369** | 0.267 |
| ATTACK | 14910 | 15175 | +265 | 0.256 |
| RECOVER | 1313 | 1087 | −226 | 0.413 |
| HAZARD | 1273 | 1097 | −176 | 0.272 |
| PHAZE | 324 | 467 | +143 | 0.133 |
| STATUS | 2044 | 2182 | +138 | 0.241 |
| PROTECT | 864 | 807 | −57 | 0.441 |
| SETUP | 957 | 946 | −11 | 0.299 |

**In one sentence: on boards the parent also faced, the fold pivots less — and it cuts the pivot
exactly where the pivot is reactive (a bad type matchup, the early game, a healthy mon, a losing
mon count) — while pivoting slightly *more* when it is ahead. It commits to the board it is on.**

### 3.1 The largest single change is invisible to every rate axis

`SWITCH→SWITCH` is the biggest flip bucket at **21.6% of all flips** — both arms want to switch,
they want *different mons*. No class-share axis can see it (the share is unchanged), which is why
the target-agreement rate is reported as a first-class number rather than left to the axes.

### 3.2 State-distribution split — read this before quoting `switch_rate`

Rows come from both arms' trajectories. Splitting the same paired delta by whose trajectory it was
measured on separates *the policy changed* from *the policy steers into boards where it differs*:

| axis | Δ pooled | Δ on PARENT's states | Δ on FOLD's states |
|---|---:|---:|---:|
| `switch\|behind_on_mons` | −0.0271 | −0.0183 | −0.0341 |
| `switch\|high_hp(>2/3)` | −0.0247 | −0.0088 | −0.0405 |
| `other_status_rate` | +0.0157 | +0.0147 | +0.0161 |
| `take_SE_attack\|SE_available` | −0.0452 | −0.0477 | −0.0351 |
| `switch\|early(turn<=8)` | −0.0378 | −0.0254 | −0.0503 |
| `switch\|losing_matchup` | −0.0385 | −0.0291 | −0.0491 |
| `switch_rate` | −0.0146 | **−0.0021** | **−0.0281** |
| `switch\|even_matchup` | −0.0200 | −0.0052 | −0.0305 |
| `switch\|ahead_on_mons` | +0.0148 | **+0.0277** | **−0.0041** |
| `switch_to_resist\|resist_avail` | −0.0154 | −0.0055 | −0.0244 |

Ten of twelve top axes keep their sign in both distributions. **Two do not, and both are
consequential:** the *aggregate* `switch_rate` reduction is near-zero (−0.2pp) on the parent's own
boards and −2.8pp on the fold's, and `switch|ahead_on_mons` flips sign outright. So the honest
statement of the aggregate finding is **conditional, not unconditional** — the fold is not a
uniformly less switch-happy policy; it is a policy whose switch reduction is largest on the boards
it steers itself into. The *conditional* reductions (losing matchup, early game, behind on mons)
survive in both distributions and are the robust part.

---

## 4. AXIS TABLE — TAUGHT (6 teams, 384 battles, 17,642 decisions)

| axis | n | parent rate | Δ (fold−parent) | 95% CI | z | realized Δ |
|---|---:|---:|---:|---|---:|---:|
| `take_SE_attack\|SE_available` | 2695 | 0.683 | **−0.0634** | [−0.0740, −0.0524] | −11.34 | −0.0261 |
| `attack_at_all\|>=2_attacks` | 4586 | 0.626 | **−0.0644** | [−0.0795, −0.0496] | −8.61 | −0.0112 |
| `attack_rate` | 15191 | 0.308 | **−0.0336** | [−0.0440, −0.0218] | −5.77 | −0.0044 |
| `take_best_damage\|>=2_attacks` | 4586 | 0.453 | **−0.0295** | [−0.0420, −0.0150] | −4.20 | +0.0422 |
| `switch_to_resist\|resist_avail` | 7361 | 0.077 | **+0.0243** | [+0.0107, +0.0388] | +3.30 | +0.0331 |
| `switch\|winning_matchup` | 2928 | 0.181 | **+0.0305** | [+0.0047, +0.0593] | +2.18 | +0.0132 |
| `switch\|opp_low_hp(<1/3)` | 1219 | 0.109 | **+0.0183** | [+0.0039, +0.0362] | +2.13 | +0.0098 |
| `switch\|we_are_boosted` | 2653 | 0.053 | **+0.0255** | [+0.0018, +0.0534] | +1.88 | +0.0392 |
| `phaze_rate` | 15191 | 0.013 | **+0.0134** | [+0.0029, +0.0302] | +1.78 | +0.0066 |
| `other_status_rate` | 15191 | 0.139 | +0.0282 | [+0.0025, +0.0669] | +1.60 | +0.0179 |
| `status_rate` | 15191 | 0.077 | −0.0057 | [−0.0131, +0.0015] | −1.54 | +0.0048 |
| `switch\|ahead_on_mons` | 6746 | 0.208 | +0.0248 | [−0.0007, +0.0616] | +1.48 | +0.0064 |
| `switch\|high_hp(>2/3)` | 10019 | 0.239 | +0.0242 | [−0.0056, +0.0603] | +1.45 | +0.0232 |
| `hazard_rate` | 15191 | 0.026 | +0.0024 | [−0.0010, +0.0056] | +1.44 | +0.0018 |
| `recover_rate` | 15191 | 0.135 | −0.0235 | [−0.0611, +0.0093] | −1.31 | −0.0457 |
| `switch\|even_matchup` | 8811 | 0.176 | +0.0257 | [−0.0090, +0.0699] | +1.27 | +0.0272 |
| `switch_rate` | 15191 | 0.224 | +0.0200 | [−0.0082, +0.0543] | +1.23 | +0.0139 |
| `switch\|late(turn>=20)` | 7928 | 0.179 | +0.0201 | [−0.0176, +0.0665] | +0.94 | +0.0288 |
| `switch\|early(turn<=8)` | 3072 | 0.324 | +0.0114 | [−0.0199, +0.0309] | +0.80 | −0.0280 |
| `switch\|behind_on_mons` | 2920 | 0.201 | +0.0147 | [−0.0335, +0.0580] | +0.62 | +0.0164 |
| `forced_repl_resists\|resist_avail` | 663 | 0.279 | −0.0119 | [−0.0554, +0.0313] | −0.53 | −0.0402 |
| `protect_rate` | 15191 | 0.017 | +0.0016 | [−0.0050, +0.0107] | +0.37 | −0.0063 |
| `setup_rate` | 15191 | 0.061 | −0.0028 | [−0.0172, +0.0124] | −0.36 | +0.0115 |
| `switch\|low_hp(<1/3)` | 1630 | 0.220 | −0.0084 | [−0.0558, +0.0335] | −0.36 | +0.0111 |
| `switch\|losing_matchup` | 3452 | 0.385 | −0.0052 | [−0.0504, +0.0357] | −0.23 | −0.0208 |

**Divergence:** argmax disagreement **0.352** (higher than untaught's 0.303, as expected) · mean KL
0.277 · flip mass involving SWITCH 0.513 · same switch target **67.7%** (vs untaught 74.8% — the
fold's switch-target *choice* moved more on the teams it was taught) · games 48.2 → 43.7 decisions
(−9.4%, against untaught's −5.2%).

**In one sentence: on the teams it taught, the fold stops clicking attacks — including the
super-effective one — and pivots instead, and it pivots into type-resists at a third again the
parent's rate.** That is the classic stall/pivot upgrade, on a taught set that is entirely
stall / semi_stall.

---

## 5. SAME OR DIFFERENT? — **DIFFERENT.**

| statistic | all 25 axes | class-share block A (9 axes) | archetype-MATCHED (untaught stall/semi_stall vs taught) |
|---|---:|---:|---:|
| cosine(untaught, taught) | **0.1425** | 0.1338 | **−0.1190** |
| permutation-null p | 0.194 | 0.384 | 0.679 |
| sign agreement | 12/25 | 4/9 | 9/25 |
| through-origin slope k | 0.101 | 0.056 | −0.106 |
| through-origin R² | 0.020 | 0.018 | 0.014 |
| ‖untaught‖ / ‖taught‖ | 0.707 | 0.418 | 0.894 |
| split-half reliability (untaught / taught) | 0.722 / 0.500 | 0.659 / 0.452 | 0.492 / 0.500 |
| **disattenuated cosine** | **0.237** | 0.245 | **−0.240** |

**Prediction 1 said the untaught fingerprint would be the taught one at lower amplitude — same
axes, smaller magnitude. Every reading refuses that.**

1. **Shape.** Cosine 0.14 is indistinguishable from a random re-assignment of the untaught vector's
   axis labels (p = 0.19), and sign agreement is 12/25 — exactly chance.
2. **The noise ceiling is not the explanation, and this is the control that matters.** Two noisy
   vectors agree less than one, so a low cosine is only evidence of a different shape if the
   vectors are individually reliable. They are: split-half reliability over teams is **0.72
   (untaught)** and **0.50 (taught)**, so a *perfectly shared* shape would read **≈0.60**. We
   measure 0.14. Disattenuated, 0.24.
3. **Amplitude is not the difference either.** ‖untaught‖ / ‖taught‖ = **0.71** — the untaught
   change is nearly as *large* as the taught one, just pointed elsewhere. A "weaker version" story
   needs a small vector along the same direction; this is a big vector along a different one.
4. **Archetype is not the confound.** v8's taught set is entirely stall/semi_stall while the
   untaught probe set is archetype-stratified, so the comparison is re-run on the six untaught
   teams whose archetype *matches*. It gets **worse**: cosine **−0.119**, disattenuated **−0.24**,
   sign 9/25. The archetype-matched untaught slice reads like the full untaught one
   (`switch|losing_matchup` −6.1pp, `switch|early` −5.9pp, `switch_rate` −2.0pp,
   `switch_to_resist` −2.4pp) — i.e. it opposes the taught vector on the same axes.

**The per-axis contrast, sorted by taught |Δ| — the anti-correlation is legible without any
statistic:**

| axis | untaught Δ | taught Δ | u / t |
|---|---:|---:|---:|
| `attack_at_all\|>=2_attacks` | +0.0065 | **−0.0644** | −0.10 |
| `take_SE_attack\|SE_available` | −0.0452 | **−0.0634** | +0.71 |
| `attack_rate` | +0.0026 | **−0.0336** | −0.08 |
| `switch\|winning_matchup` | +0.0071 | +0.0305 | +0.23 |
| `take_best_damage\|>=2_attacks` | −0.0080 | −0.0295 | +0.27 |
| `other_status_rate` | +0.0157 | +0.0282 | +0.56 |
| `switch\|even_matchup` | −0.0200 | +0.0257 | −0.78 |
| `switch_to_resist\|resist_avail` | −0.0154 | +0.0243 | −0.64 |
| `switch\|high_hp(>2/3)` | −0.0247 | +0.0242 | −1.02 |
| `switch_rate` | −0.0146 | +0.0200 | −0.73 |
| `switch\|behind_on_mons` | −0.0271 | +0.0147 | −1.84 |
| `switch\|early(turn<=8)` | −0.0378 | +0.0114 | −3.32 |
| `switch\|losing_matchup` | −0.0385 | −0.0052 | **+7.42** |

**Exactly one axis behaves the way prediction 1 required** — `take_SE_attack|SE_available`, which
falls on both slices with u/t = 0.71, i.e. the taught shape at 71% amplitude. It is the one piece
of content that looks *diluted* rather than *different*, and it is the one to carry forward. Every
other large axis either flips sign or moves at a ratio the "same content, less of it" story cannot
produce (`switch|losing_matchup` at 7.4× the taught magnitude is not a dilution).

---

## 6. ATTRIBUTING THE UNTAUGHT GAIN — what co-occurs with the +5.66pp

**Read as co-occurrence, never causation.** The behavioural delta and the win-rate delta are
computed on the same battles, so a common cause (a matchup that simply admits more pivoting *and*
more winning) reproduces any correlation here without any behaviour causing any winning.

**Per team (16 points, Spearman, team bootstrap):** the top correlates are all switch-conditional
and all point the same way — the teams where the fold cut reactive switching hardest are the teams
that gained most.

| axis | ρ (per-team Δbehaviour vs Δwin-rate) | 95% CI |
|---|---:|---|
| `switch\|winning_matchup` | **+0.565** | [+0.214, +0.818] |
| `switch\|early(turn<=8)` | **−0.518** | [−0.801, −0.055] |
| `switch\|losing_matchup` | −0.447 | [−0.784, +0.042] |
| `switch\|high_hp(>2/3)` | −0.444 | [−0.777, +0.038] |
| `hazard_rate` | −0.421 | [−0.757, +0.011] |
| `other_status_rate` | +0.400 | [−0.153, +0.774] |

Only the first two clear zero, at n=16. The *amount* of change is not what predicts the gain:
per-team argmax-disagreement vs win-rate delta is **ρ = −0.13** — a team where the fold behaves
very differently is not thereby a team where it wins more. **It is the direction, not the dose.**

**Per cell (128 team×opponent cells, team-clustered):** every |ρ| ≤ 0.20 and only
`switch|late(turn>=20)` (−0.195 [−0.356, −0.029]) excludes zero. The signal lives at the team
level and is not resolvable cell by cell at 4 games per cell.

**Battle-level (TAUGHT slice only, for now).** Because both arms play the same seed at the same
game index, every game is a paired trial with four outcomes, and the win-rate gain is
`(FLIP_WIN − FLIP_LOSS)/N` *by identity*. On the taught slice: 192 paired games, **FLIP_WIN 63 ·
FLIP_LOSS 19 · BOTH_WIN 67 · BOTH_LOSS 43 ⇒ +0.2292**, which is the +22.92pp of §2.1 recovered from
the other direction. Contrasting the fold's behavioural delta *inside* the games it flipped to a
win against the games both arms lost:

| axis | Δ in FLIP_WIN games | Δ in BOTH_LOSS games | difference |
|---|---:|---:|---:|
| `switch\|opp_low_hp(<1/3)` | +0.0327 | −0.0336 | **+0.0663** |
| `switch\|behind_on_mons` | +0.0268 | −0.0335 | +0.0603 |
| `switch\|winning_matchup` | +0.0352 | −0.0026 | +0.0378 |
| `take_SE_attack\|SE_available` | −0.0460 | −0.0835 | +0.0375 |
| `switch\|high_hp(>2/3)` | +0.0178 | −0.0143 | +0.0321 |
| `switch\|late(turn>=20)` | +0.0222 | −0.0093 | +0.0315 |
| `forced_repl_resists\|resist_avail` | −0.0170 | +0.0142 | −0.0312 |
| `switch_to_resist\|resist_avail` | +0.0371 | +0.0104 | +0.0267 |

Six of the eight are switch axes and the pattern is uniform: **the games the taught fold flipped to
a win are the games in which it out-pivoted the parent; the games both arms lost are the ones where
it pivoted no more, or less, than the parent would have.** Same instrument, opposite sign from the
untaught story — §5's verdict arriving again by a different route. (`take_SE_attack` is the
exception that confirms it: the fold clicks the super-effective move less in *both* buckets, and
much less in the losses.)

**Battle-level (UNTAUGHT) — the same instrument, and it comes back FLAT.** The per-game outcome
vectors were added to the recorder after the untaught pass had already run, so all 256 untaught
cells were **replayed in `--outcomes-only` mode** to supply them. Greedy play on a fixed seed is
deterministic, and that is checked rather than assumed: **256 of 256 replayed cells report the
identical win count, 0 disagreeing** (`replay_determinism` in the JSON), so the join is sound.
512 paired games: **FLIP_WIN 126 · FLIP_LOSS 97 · BOTH_WIN 96 · BOTH_LOSS 193 ⇒ +0.0566**, the
+5.66pp recovered by identity.

| axis | Δ in FLIP_WIN games | Δ in BOTH_LOSS games | difference |
|---|---:|---:|---:|
| `forced_repl_resists\|resist_avail` | +0.0164 | −0.0299 | +0.0463 |
| `switch\|ahead_on_mons` | −0.0054 | +0.0283 | −0.0337 |
| `switch\|early(turn<=8)` | −0.0278 | −0.0521 | +0.0243 |
| `switch\|low_hp(<1/3)` | +0.0154 | −0.0052 | +0.0206 |
| `switch\|late(turn>=20)` | −0.0304 | −0.0105 | −0.0199 |
| `take_SE_attack\|SE_available` | −0.0426 | −0.0298 | −0.0128 |
| `switch\|high_hp(>2/3)` | −0.0357 | −0.0272 | −0.0085 |
| `switch\|losing_matchup` | −0.0502 | −0.0424 | −0.0078 |

**The core untaught axes look the SAME in the battles the fold flipped to a win and in the battles
both arms lost** — `switch|losing_matchup` −5.0 vs −4.2pp, `switch|high_hp` −3.6 vs −2.7,
`take_SE_attack` −4.3 vs −3.0. Against the taught slice's table above (differences of +0.03 to
+0.07, uniformly switch-directional), this is a clean asymmetry:

> **On taught teams the fold's wins are localisable — it wins the games in which it out-pivots the
> parent. On untaught teams they are not: the behavioural shift is broad and roughly uniform across
> won and lost games, so no small set of decisive interventions carries the +5.66pp.**

Combined with §6's per-team result — direction predicts the gain, dose does not — the untaught
content reads as a *global re-weighting of when to pivot*, not as a handful of saved games. That is
consistent with a gift that generalises, and it is the third independent way this probe separates
the two slices.

---

## 6b. THE GEN-ERA ARM — what v8 changed that a fold which did NOT gift did not

The comparison the mission called the real prize, and it ran: the **rev-3 fold**
(`ai_v9_70_R3ACTION_0828`) against **its own parent** (`ai_v9_59_R2ACTION_0827`), reference
opponent `ai_v9_29_rev1_0823` (an ancestor of both), on probe Q's 8 untaught teams + one taught
team per rev-3 F6 teacher, same 8 fixed opponents, same CRN construction, current architecture,
rust bridge. **896 battles · 39,194 dual-scored decisions.** Its win rates reproduce the era's
known result: **untaught −0.39pp** (probe Q: −0.75pp — no gift) and **taught +4.69pp** (against
v8's +22.92pp on the same instrument).

### The gen fold changes untaught behaviour MORE than v8 did, and wins nothing for it

‖untaught behavioural vector‖ is **0.177 for gen vs 0.093 for v8 — 1.9× larger** — while the
win-rate delta goes the other way (−0.39pp vs +5.66pp). **"The fold changed how it plays on
untaught teams" is not what a gift is.** Whatever separates a gift from a robbery is a *direction*,
not an amount. (That is the same lesson §6 found inside v8 alone: per-team argmax disagreement does
not predict per-team gain, ρ = −0.13.)

### The two untaught fingerprints, side by side

cosine(v8-untaught, gen-untaught) = **0.113**, permutation-null p = **0.50** — dead centre of the
null — sign agreement 14/25, both vectors reliable (split-half 0.72 / 0.74, ceiling 0.733),
disattenuated **0.154**. Sorted by the gen fold's own magnitude:

| axis | **v8** untaught Δ (**+5.66pp gift**) | **gen** untaught Δ (**−0.39pp, no gift**) | |
|---|---:|---:|---|
| `attack_at_all\|>=2_attacks` | +0.0065 | **+0.0752** | ⟂ |
| `take_best_damage\|>=2_attacks` | −0.0080 | **+0.0573** | **opposed** |
| `switch_to_resist\|resist_avail` | −0.0154 | **−0.0561** | shared |
| `switch\|ahead_on_mons` | **+0.0148** | **−0.0552** | **opposed** |
| `take_SE_attack\|SE_available` | **−0.0452** | **+0.0505** | **opposed** |
| `switch\|winning_matchup` | +0.0071 | **−0.0462** | **opposed** |
| `forced_repl_resists\|resist_avail` | −0.0197 | −0.0443 | shared |
| `switch\|low_hp(<1/3)` | **+0.0171** | **−0.0440** | **opposed** |
| `switch\|losing_matchup` | **−0.0385** | **−0.0420** | **shared, near-identical** |
| `attack_rate` | +0.0026 | +0.0339 | ⟂ |
| `switch\|high_hp(>2/3)` | −0.0247 | −0.0304 | shared |
| `switch_rate` | −0.0146 | −0.0293 | shared |
| `switch\|late(turn>=20)` | −0.0033 | −0.0289 | ⟂ |
| `switch\|we_are_boosted` | +0.0097 | −0.0237 | opposed |
| `switch\|early(turn<=8)` | **−0.0378** | −0.0220 | shared |
| `switch\|even_matchup` | −0.0200 | −0.0191 | shared |
| `switch\|behind_on_mons` | **−0.0271** | −0.0150 | shared |
| `switch\|opp_low_hp(<1/3)` | +0.0092 | −0.0137 | opposed |
| `other_status_rate` | **+0.0157** | −0.0030 | opposed |

**The structure is not "different everywhere" — it splits cleanly, and the split is the finding.**

- **SHARED (so NOT what makes a gift):** both folds cut switching, and both cut it in the *bad and
  neutral* spots — `switch|losing_matchup` −3.85 vs −4.20pp is almost the same number, and
  `switch|even_matchup`, `switch|high_hp`, `switch|early` and aggregate `switch_rate` all move the
  same way. "Stop pivoting out of a bad matchup" is present in the fold that gifted **and** in the
  fold that did not. It cannot be the discriminator.
- **OPPOSED (the candidate discriminator):** every axis about *what to do when you are winning or
  when precision is available* flips. The gen fold cuts the opportunistic pivots
  (`switch|ahead_on_mons` −5.5, `switch|winning_matchup` −4.6, `switch|low_hp` −4.4, all where v8
  went **up**) and clicks harder (`take_SE_attack` **+5.1** vs v8's **−4.5**; `take_best_damage`
  +5.7 vs −0.8; `attack_at_all` +7.5 vs +0.7).

**In one sentence: the fold that did not gift got GREEDIER on untaught teams — more
maximum-damage clicks, fewer pivots even when ahead — while the fold that did gift got more
SELECTIVE: it cut the reactive pivot but kept and slightly raised the opportunistic one, and it
clicked the super-effective move less.** The shared "switch less in bad spots" component is common
to both and therefore explains neither.

### What the gen arm does NOT support

Its own **taught** vector is not usable for a shape comparison: split-half reliability **0.077**,
i.e. essentially zero. The rev-3 fold's taught-side behavioural change is too small relative to
between-team variance to have a stable direction at 6 teams — unsurprising given its taught win-rate
delta is +4.69pp against v8's +22.92pp. So `shape` and `shape_archetype_matched` are present in the
JSON for the gen family and **must not be quoted**; the gen arm contributes the untaught vector and
nothing else.

**Three honest limits on the cross-family comparison.** (1) The team sets differ — v8's 16
archetype-stratified pool teams vs probe Q's 8 curated-32 teams — so this is not a controlled
contrast, only a comparison of directions in a shared, architecture-free basis (which is the whole
reason the basis was built that way). (2) The reference opponents and architectures differ by
construction; only the axis definitions are held fixed. (3) The gen untaught parent sits at
**0.508** win rate against **0.377** for v8's, so the two folds had different headroom — the
scorecard's own floor/ceiling confound, here in the opposite direction from probe Q's.

---

## 7. Predictions, scored

1. **P1 — "the untaught fingerprint is a WEAKER VERSION of the taught one (same axes, smaller
   magnitude)": FAIL, and not marginally.** Cosine 0.14 against a 0.60 reliability ceiling,
   permutation p = 0.19, sign agreement at chance, R² 0.02, and a norm ratio of 0.71 that rules out
   "smaller" as the difference. The archetype-matched control makes it *negative*. The one axis
   that behaves as predicted is `take_SE_attack|SE_available` (u/t = 0.71).
2. **P2 — "SWITCH-related decisions carry the largest share of the untaught behavioural change":
   PASS.** Six of the seven axes with |z| > 2.8 are switch axes; **51.0% of all argmax flips involve
   a SWITCH on one side or the other**; SWITCH has the largest class net flow (−445, larger in
   absolute terms than any other class) and the largest flip-out rate among classes with meaningful
   n (0.373); and `SWITCH→SWITCH` is the single biggest flip bucket (21.6%). The per-team
   attribution's top six correlates are switch axes without exception. **One qualification the
   prediction did not anticipate: the largest single component is a change of switch TARGET, not of
   switch RATE** — an axis no class-share statistic can see, and one worth building into the next
   instrument rather than discovering again.

---

## 8. What this changes

- **The `(i)/(ii)` fork gets evidence for (ii).** The gen-era fleets are shape-selected on
  **taught-side** fold quality. v8's untaught gift is measurably *not* made of taught-side content
  — on the decisive axes it is the opposite behaviour. A metric family that scores the taught
  vector cannot rank fleets by the untaught one. This does not settle (i) (maturity remains the
  prime suspect by elimination) — it says the *measurement* branch is live on its own evidence.
- **"Transfer is local" needs re-wording.** Probe P framed the ~4.8:1 taught:untaught ratio as
  locality that "changed the SIGN of what leaks out". At the behavioural level there is no leak to
  speak of: the untaught change is 71% of the taught change's magnitude and roughly orthogonal to
  it. Whatever the fold did on the untaught teams, it is a *second* thing it learned, not a spillover
  of the first.
- **A concrete, portable candidate — and the gen arm already narrowed it.** The obvious reading of
  §3 alone is "stop pivoting reactively and commit". **§6b refutes that as the discriminator**: the
  non-gifting rev-3 fold does the same thing, at the same size, and gains nothing. What survives as
  the candidate is the *selectivity* half — **cut the reactive pivot while keeping or raising the
  opportunistic one (ahead on mons, at low HP, opponent nearly dead), and do NOT become a
  greedier maximum-damage clicker**. That is architecture-free and testable on the current
  generation directly (§9).
- **A generic caution the two arms establish jointly.** Behavioural-change *magnitude* is not a
  progress signal. Within v8, per-team disagreement does not predict per-team gain (ρ = −0.13);
  across folds, the one that changed untaught behaviour most gained least. Any future
  "the fold moved the policy a lot" reading needs a direction attached before it means anything.
- **It does not contradict M1, and the word "axis" is doing different work in the two.**
  `axis_split_taught_untaught_2026-08-31.md` measures whether fleet SHAPES *rank the same way* on
  the taught-side and untaught-side **win-rate** meters, and finds ρ = +1.00 — its "axes" are two
  scoreboards. This probe's "axes" are behavioural coordinates, and it finds the taught and
  untaught **content** to be near-orthogonal. Both can hold at once, and together they say
  something sharper than either: *the same fleet-shape ordering scores two different kinds of
  change.* M1 also rules out shape as the recovery route for v8's gift, which raises the value of
  a content-side account like this one rather than lowering it.
- **The `SWITCH→SWITCH` finding is a standing instrument gap.** A fifth of all behavioural change
  between two policies here is a change of switch target that every rate-based behavioural metric
  in this programme would score as zero.

## 9. What to do next (not run here)

1. **Test the SELECTIVITY contrast on the current generation.** Not "reduce reactive pivoting" —
   §6b shows that is shared with a fold that gained nothing. The discriminating hypothesis is the
   *opposed* set: `switch|ahead_on_mons`, `switch|winning_matchup`, `switch|low_hp`,
   `take_SE_attack`, `take_best_damage`. Measure those five against win rate across the running
   fleet's folds — no v8 code required, the axes are architecture-free — and if they predict, the
   lever is a shaping/objective target rather than a distillation one.
2. **Split the switch-target axis out.** Add "which mon" as a first-class axis family (resist,
   HP, role, revealed-threat coverage) before the next behavioural comparison.
3. **Replicate on a second gifting fold if one is ever produced.** n=1 gifting fold is the
   structural limit of this measurement; the gen arm supplies an n=1 *control*, not a second
   gift. See the caveats.

---

## Caveats (read before quoting)

1. **n = 1 gifting fold, n = 1 non-gifting control.** Every conclusion about "what a gift is made
   of" rests on one fold (`ai_v8_14`) against one parent, with one contrast fold
   (`ai_v9_70_R3ACTION`) on the other side. The taught-vs-untaught split is internal to v8 and is
   therefore sound; the shared-vs-opposed split of §6b is a two-point comparison across eras and is
   **a hypothesis-narrowing result, not a verdict**.
2. **Co-occurrence is not causation, and §6 says so twice.** The behavioural deltas and the
   win-rate deltas are the same battles. Nothing here demonstrates that reducing reactive
   switching *caused* the gain.
3. **The aggregate `switch_rate` reduction is state-conditional** (§3.2): −0.2pp on the parent's own
   boards, −2.8pp on the fold's. Quote the *conditional* axes (losing matchup / early / behind),
   which survive both distributions; do not quote the unconditional switch rate as a property of
   the policy.
4. **Greedy, not stochastic.** Both sides play `stochastic=False`, matching probe P and the era's
   `vs_ext` eval regime — but v8's actual training and the admission games were stochastic. This is
   a fingerprint of greedy play by these two networks.
5. **Sub-sample of probe P.** 4 games per cell of probe P's 30, chosen as the seed-sequence prefix
   so the battles are literally probe P's. The win-rate reproduction (+5.66 vs +5.42) is the check
   that it did not drift; it is not an independent re-measurement.
6. **The matchup stratifier is a type proxy** (best STAB effectiveness each way) and knows neither
   side's moveset. "Losing matchup" means "their types beat ours", not "they have a move that
   kills us".
7. **`switch_to_resist` and `forced_repl_resists` use the same type proxy on the switch target**,
   so they answer "does the incoming mon resist their types", not "is this the right pivot".
8. **The taught slice is 6 teams.** Its split-half reliability is 0.50 — the lower of the two, and
   the reason the disattenuated cosine is quoted beside the raw one rather than instead of it.
9. **The era-pin trace validation inherits a known defect.** `mask = (recorded_logits > −1e8)`
   returns ALL 11 actions legal on those traces (the phantom-legality class in
   `designs/learning/vacuous_tests_and_guards.md`), so the *argmax-agreement* figure in §2.1 is
   computed under an all-legal mask and is looser than reality. The logit and V correlations are
   unaffected, and **the axis tables never touch recorded logits** — they use the live mask from
   the live battle.
10. **A cross-shard JOIN-KEY defect was found and fixed mid-analysis; recorded because the class
    is the programme's most expensive one.** The bridge numbers battles `battle-gen3ou-N` from 1
    **per process**, so two shards emit identical tag strings for different battles. Every
    tag-keyed join was therefore silently mislabelling roughly half its rows: the battle count read
    512 instead of 1,024, the mean game length read exactly double, and the taught battle-level
    attribution table (§6) was wrong — its numbers changed materially on the fix. The axis tables
    and the shape verdict are **untouched**: nothing in them joins on a tag. The analyzer now keys
    on `(team, opp, arm, tag)`, which is unique because shards partition the team list.
11. **Operational note, recorded because it nearly cost the taught slice.** A first taught pass was
    launched twice concurrently by two supervisors writing to one gzip sink; the interleaved stream
    was discarded and the slice re-run clean from an empty file. The reported taught numbers come
    from a single, uncontended pass (96/96 cells, no duplicate cell records — the analyzer's
    `replay_determinism` block reads `duplicate_cells: 0`).

---

## Reproduce

```bash
# the battle pass — v8 family, from an era-pinned worktree at b13b30b
git worktree add --detach /tmp/probeP_v8era b13b30b289c5eaba136a930a4ab63451e209fbe5
PYTHONPATH=/tmp/probeP_v8era/src nice -n 15 python \
  designs/research_state/measurements/v8_fold_behavioral_fingerprint_probe.py \
  --family v8 --games 4 --opps 8 --untaught 16 --taught 0 --shard 0/2 --out /tmp/m4/run
# ... and --untaught 0 --taught 6 --out /tmp/m4/runT for the taught controls

# the gen-era arm — current tree, rust bridge
export PYTHONPATH=$PYTHONPATH:src
nice -n 15 python designs/research_state/measurements/v8_fold_behavioral_fingerprint_probe.py \
  --family gen --impl rust --games 4 --opps 8 --teams-json <selection.json>

# the analysis — NO BATTLES, straight off the banked rows (~2 min), which is how the committed
# JSON was produced. `--label` selects the family; the cell file carries both, filtered by `fam`.
M=designs/research_state/measurements/v8_fold_behavioral_fingerprint_2026-08-31
python designs/research_state/measurements/v8_fold_behavioral_fingerprint_analyze.py \
  --rows "${M}_rows_v8.jsonl.gz" --cells "${M}_cells.jsonl" --label v8 --out "$M"
python designs/research_state/measurements/v8_fold_behavioral_fingerprint_analyze.py \
  --rows "${M}_rows_gen.jsonl.gz" --cells "${M}_cells.jsonl" --label gen --out "$M" \
  --merge-into "$M.json"
```
