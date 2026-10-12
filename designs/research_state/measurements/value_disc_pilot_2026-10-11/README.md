# The value-discrimination pilot (2026-10-11): oracle-world decision improvement on the end-state and production seeds

**Status: Phases 1, 2, 2b and 3 RUN AND READ (2026-10-11): Phase 1 (oracle world) §6; Phase 2 / 2b (honest worlds)
§7; Phase 3 (an offline per-action value head on oracle labels) §12. Phase 2c (worlds from the network's own MOVE
belief) REGISTERED, §10 (Amendment 6), running.** This README also holds the pre-registration (a copy of
`designs/endstate/design_value_discrimination.md` §7 as amended by §7.10 Amendments 1–7): §2–§5, §8, §10, §11.

## 1. In plain language

At a decision, a one-ply search tries each legal action, estimates how good the position after it is, and plays
the best. The question is which estimate is worth its cost:

- the CRITIC (the network's own win-probability guess for the position after the action), which is nearly free;
- ROLLOUTS (play the game out K times from the position after each action and count wins), which are expensive;
- SHORT rollouts (play only 1, 2 or 4 turns, then ask the critic), which are in between;
- a LUCK-CORRECTED rollout (subtract how lucky the first turn's dice were, using the critic as the yardstick).

And which ACTING RULE turns those estimates into a move: take the best estimate (argmax), mix the estimate with
the policy's own preference (prior-aware, as Gumbel AlphaZero does), spend the playout budget on the policy's
favourites in rounds (sequential halving), or the old rule that switched away from the policy's pick only when a
difference was statistically resolved (2 standard errors).

The yardstick is DECISION IMPROVEMENT: how much more often the game is won, in win probability, if we play the
rule's pick instead of the policy's own top choice, judged on 24 rollouts per action that no scorer ever saw.

The big caveat: every rollout here plays from the TRUE battle, knowing the opponent's real team and sets (the
ORACLE world). The critic does not know them. So a rollout leaf's edge over the critic is partly hidden
information that no real player has.

## 2. What is fixed (the registration; Amendment 1 of the design doc's §7.10)

| item | value |
|---|---|
| checkpoints | **end2** = `models/rb_es_end2_s2001/final_model.zip` (15.05M, pin `62085750`, tag `closing-test-pend2`) and **prod** = `models/rb_es_prod_s2001/final_model.zip` (15.05M, pin `95d014fa`). Each is loaded, encoded and played at its OWN pin's code (a detached checkout; `run_at_pin.sh`), CPU only, through the strict loader; `check_winprob` passes on both |
| states | each checkpoint's OWN eval traces: the last three cycles (its 10M / 12M / 14M snapshots, recorded at the pin), EVERY trace of every roster opponent (end2 661 traces, prod 679). Gate ① (re-encoding reproduces every recorded obs row byte-equal, masks equal, actions equal): end2 24,036 / 24,036 decisions, prod 23,392 / 23,392. The obs clock decodes to the protocol's turn on every decision (same counts) |
| eligible | a free move round, turn 2–40, ≥ 3 legal actions: end2 16,693, prod 16,099 |
| strata | the FINAL checkpoint's top-2 masked-logit gap at the root, quintiles over the eligible pool: end2 edges 0.265 / 0.605 / 1.071 / 1.874; prod 0.324 / 0.768 / 1.407 / 2.582. A state within 1e-6 of an edge, or with a top-2 tie, is excluded (rule 8; end2 1, prod 0) |
| sample | 120 per quintile = 600 per checkpoint, at most 2 per battle, in `sha256("value_disc_pilot_2026:" + id)` order (end2 417 battles, prod 416). Mean legal actions 7.46 / 7.39. Traced outcome: end2 321 L / 262 W / 17 D; prod 317 / 266 / 17 (the trace quota is loss-enriched) |
| options | ALL legal actions; `top1` = the final checkpoint's masked-logit argmax at the root (lowest index on a tie, never reached: ties excluded) |
| continuation | CP greedy on both sides to the end (`play_out`'s request: production stall, `max_turns` 999), rooted at the decision's command index; the other side's open root decision answered by CP's greedy policy identically across siblings. +1 win / −1 loss / 0 tie or truncated. **World: ORACLE** |
| seeds | `truth.turn_seeds(id, 48)`, shared by every action: **0–15 leaf** (every scorer's playouts), **16–39 truth** (24; no scorer ever reads them), **40–47 the critic's one-ply successors** (`qhat.one_ply`, never played on) |
| value trace | for every (action, seed): CP's V at every searched-side decision, with its turn (decoded from the obs clock) and a switch-only flag; the terminal |
| ties | an argmax tie (\|Δscore\| < 1e-6) breaks toward the policy's order (top1 first, then by logit) |
| CIs | battle-clustered bootstrap, 2,000 draws, `numpy.random.default_rng(20261011)`, percentile 95 %, every contrast paired on the same draws; per checkpoint |

## 3. The scorers (per action a, all on the win scale [0, 1])

| scorer | definition |
|---|---|
| CRITIC | Q̂(a) = mean over seeds 40–47 of the critic's V at the searched side's next decision (a branch that ends first scores its terminal) |
| LEAF-K, K = 1, 2, 4, 8, 16 | the mean outcome over seeds 0..K−1 |
| TRUNC-d-K, d = 1, 2, 4; K = 1, 4, 8, 16 | the mean over seeds 0..K−1 of X_d = V at the searched side's first decision whose turn ≥ T + d (T = the root's turn), or the terminal if the game ended first |
| CV-K (AIVAT-style, branch turn), K = 1, 2, 4, 8, 16 | LEAF-K(a) − β·(mean over seeds 0..K−1 of X₁′(a, j) − Q̂(a)), β = 1, where X₁′ = V at the searched side's FIRST decision after the root (the quantity `one_ply` captures, terminal if ended): the first turn's dice luck as the critic sees it, subtracted against the critic's own independent estimate of its mean (seeds 40–47). Unbiased for LEAF-K's mean for any fixed β. The per-turn AIVAT form (extra dice samples at every turn) is DEFERRED |
| MIX-K, K = 1, 4, 8 | ½·CRITIC + ½·LEAF-K (AlphaGo's λ = 0.5) |

## 4. The acting rules

| rule | definition |
|---|---|
| ARGMAX | the scorer's argmax over all legal actions |
| PRIOR-k | the argmax over the policy's top-k (k = 2, 3) |
| GUMBEL (prior-aware; primary of its kind) | Gumbel MuZero's final selection WITHOUT noise, `mctx` defaults: argmax of logit(a) + (c_visit + N_max)·c_scale·q̃(a), c_visit = 50, c_scale = 0.1, q̃ = the scorer's values min-max rescaled to [0, 1] over the legal actions (all equal ⇒ 0), N_max = the per-action sample count (K; 8 for the critic). Logits = the root's raw masked logits |
| PUCT (descriptive) | argmax of s(a) + 1.25·π(a)·√(n·K)/(1 + K), π the masked softmax, n the legal count |
| SH-m (sequential halving, Gumbel AlphaZero's allocation, no noise), m = 4 and m = all | budget B = n·K playouts for K = 1, 2, 4, 8 (matched to LEAF-K's spend); candidates = the policy's top-m; R = ⌈log₂ m⌉ rounds; in a round each survivor gets max(1, ⌊B / (R·\|S\|)⌋) new playouts (its leaf seeds in order), capped at 16 per action; after each round keep the top ⌈\|S\|/2⌉ by the GUMBEL score on the samples so far; the pick is the last survivor (or the GUMBEL argmax of the last pair). The SPENT budget is reported beside B |
| GATE-2SE (the old rule) | switch from top1 to LEAF-K's argmax a′ only if K ≥ 4 and the paired mean d = mean_j (o(a′, j) − o(top1, j)) over seeds 0..K−1 satisfies d ≥ 2·SE, SE = max(sd(d)/√K, 0.5/K) on the win scale; K = 4, 8, 16 |

## 5. Metrics and registered decision rules (per checkpoint)

**D1 (PRIMARY): IMP = mean over states of T(â) − T(top1)**, T = the mean outcome over the 24 truth seeds on the
win scale. Reported for every scorer × rule, by gap quintile, by unrevealed opponent slots (0–1 / 2–3 / 4–5), by
phase (turn 2–10 / 11–25 / 26–40), by the trace's outcome; raw AND re-weighted to the natural outcome rate of each
(cycle, opponent) (weight = battles of that outcome played / traced); the raw one decides. The oracle bound
max_a T(a) − T(top1) is reported with its winner's-curse stamp. **Cost** per decision: CPU-s from the measured
forward rows (both sides) each scorer's playouts need, at the state's measured CPU-s per row; the critic's
measured one-ply CPU.

A bound within ±0.0005 of its boundary counts as not clear (rule 8).

| rule | contrast (paired) | outcomes |
|---|---|---|
| **R-LEAF (primary)** | IMP(LEAF-K8) − IMP(CRITIC), ARGMAX | LEAF-AHEAD: lower > +0.0005 · CRITIC-EQUIVALENT: CI inside ±0.01 · NOT DETECTED |
| R-TRUNC | smallest d ∈ {1, 2, 4} with IMP(TRUNC-d-8) − IMP(LEAF-K8) lower > −0.01 AND IMP(TRUNC-d-8) − IMP(CRITIC) lower > +0.0005 | that d, or NONE |
| R-K | IMP(LEAF-K8) − IMP(LEAF-K4) | SATURATING: upper < +0.01 · STILL CLIMBING: lower > +0.0005 · UNRESOLVED |
| R-PRIOR (added) | IMP(GUMBEL∘LEAF-K4) − IMP(ARGMAX∘LEAF-K4) | PRIOR HELPS: lower > +0.0005 · PRIOR HURTS: upper < −0.0005 · NOT DETECTED |
| R-SH (added) | IMP(SH-4 at B = n·2) − IMP(ARGMAX∘LEAF-K2) | SH HELPS / SH HURTS / NOT DETECTED, as R-PRIOR |
| R-PAIR | ρ(1) lower ≥ 0.30 AND ρ(terminal) upper < 0.20 (all pairs, terminal and X_d from the leaf seeds) | ALIGNMENT-LOSS SIGNATURE ⇒ build S1 · NO SIGNATURE |

Everything else (CV-K, MIX-K, PRIOR-k, PUCT, GATE-2SE, the K curve to 16, M1–M3) is descriptive.

**Predictions** (the doc's §7.6, carried over; written before measurement): IMP(LEAF-K8) +0.02 to +0.06,
IMP(CRITIC) −0.01 to +0.02, R-LEAF LEAF-AHEAD; R-K UNRESOLVED or STILL CLIMBING; R-TRUNC d = 4; ρ(terminal)
0.05–0.12, R-PAIR NO SIGNATURE. Added: R-PRIOR NOT DETECTED or PRIOR HELPS at K = 4; R-SH SH HELPS (a guess).

**Size rule.** The throughput pilot (4 end2 states, cost only, no outcome printed) measured 0.060 CPU-s per
playout, ~15 CPU-s per state: ≈ 2.5 CPU-hours per checkpoint, far under the 12-hour cap. No state is dropped.

## 6. Phase 1 results (ORACLE world; read 2026-10-11 15:17; full tables `result.md`, numbers `result.json`)

**In plain language.** On both checkpoints' own states, playing every legal action out with the network and
picking the best beats the network's own pick by about **7–8 points of win probability per decision** at 8
rollouts per action, and the curve is still climbing at 16. The critic alone buys 1–3 points. Short rollouts
(4 turns, then the critic) buy about half of the full rollout's value at a seventh of its cost; 1–2 turns buy
nothing over the critic. Trusting the policy's prior in the acting rule (Gumbel's rule) LOSES 1–3 points: where
the rollouts disagree with the policy, the rollouts are usually right. Spending the budget on the policy's top
choices (sequential halving) is no better than spreading it evenly. Pairing the dice shares real luck for one
turn and loses it after (so a per-turn reseed would pay). All of this is in the ORACLE world: Phase 2 measures
how much survives honest information.

All 1,200 states played (no refusal); 417 / 416 battles. IMP on the win scale, [95 % battle-clustered], raw (the
registered read) and re-weighted to the natural outcome rate; CPU-s per decision on this box's contended CPU
forward (6 logical CPUs; the forward is ~95 % of it).

| scorer (ARGMAX over all legal) | end2 IMP | end2 CPU-s | prod IMP | prod CPU-s |
|---|---|---|---|---|
| CRITIC (one-ply, 8 seeds) | +0.026 [+0.014, +0.039] | 0.66 | +0.012 [+0.000, +0.023] | 0.44 |
| LEAF-K1 | +0.035 [+0.024, +0.046] | 2.4 | +0.026 [+0.014, +0.039] | 1.4 |
| LEAF-K2 | +0.054 [+0.041, +0.067] | 4.9 | +0.037 [+0.024, +0.051] | 2.9 |
| LEAF-K4 | +0.069 [+0.055, +0.083] | 9.8 | +0.058 [+0.044, +0.072] | 5.7 |
| **LEAF-K8** | **+0.081 [+0.066, +0.095]** | 19.6 | **+0.071 [+0.057, +0.087]** | 11.5 |
| LEAF-K16 | +0.098 [+0.084, +0.113] | 39.0 | +0.082 [+0.067, +0.097] | 23.0 |
| TRUNC-1-K8 / TRUNC-2-K8 | +0.028 / +0.040 | 0.7 / 1.3 | +0.014 / +0.015 | 0.4 / 0.9 |
| TRUNC-4-K1 / K4 / K8 / K16 | +0.027 / +0.052 / +0.051 / +0.052 | 0.33 / 1.3 / 2.6 / 5.2 | +0.028 / +0.037 / +0.037 / +0.041 | 0.21 / 0.85 / 1.7 / 3.4 |
| MIX-K1 / MIX-K8 (½ critic + ½ leaf) | +0.049 / **+0.087** | 3.1 / 20.3 | +0.034 / +0.076 | 1.9 / 11.9 |
| CV-K8 (AIVAT-style, branch turn) | +0.078 | 20.3 | +0.071 | 11.9 |
| ORACLE upper bound (winner's curse) | +0.152 | — | +0.132 | — |

| rule (registered) | contrast | end2 | prod | verdict (both) |
|---|---|---|---|---|
| **R-LEAF (primary)** | LEAF-K8 − CRITIC | +0.055 [+0.040, +0.068] | +0.059 [+0.045, +0.075] | **LEAF-AHEAD** |
| R-TRUNC | TRUNC-4-K8 − LEAF-K8 / − CRITIC | −0.029 [−0.042, −0.017] / +0.025 [+0.012, +0.038] | −0.035 / +0.025 | **NONE** (no depth ≤ 4 keeps the full leaf's value within 1 point) |
| R-K | LEAF-K8 − LEAF-K4 | +0.012 [+0.004, +0.020] | +0.014 [+0.006, +0.021] | **STILL CLIMBING** (K16 − K8 +0.017 / +0.010, clear) |
| R-PRIOR | GUMBEL − ARGMAX at LEAF-K4 | −0.013 [−0.021, −0.005] | −0.022 [−0.031, −0.013] | **PRIOR HURTS** (at K8: −0.022 / −0.031) |
| R-SH | SH-4 (B = n·2) − ARGMAX LEAF-K2 | −0.005 [−0.015, +0.006] | −0.002 [−0.012, +0.008] | **NOT DETECTED** (at B = n·4 and n·8 SH-4 is WORSE: −0.011…−0.027) |
| R-PAIR | ρ(1) / ρ(terminal) | 0.44 [0.37, 0.52] / 0.062 [0.051, 0.076] | 0.53 [0.45, 0.62] / 0.065 [0.052, 0.079] | **ALIGNMENT-LOSS SIGNATURE** ⇒ the registered rule says build S1 (per-turn reseed) |

- **Validity:** the derangement VRF contains 1 (end2 0.999, prod 0.998); the CV's two estimates of the first
  decision's V agree (mean difference −0.0003 [−0.0011, +0.0005] / 0.0000 [−0.0008, +0.0008]); the opponent's root
  decision was open (answered greedily) on 98.7 % / 100 % of roots.
- **Old acting rule:** GATE-2SE at K8 keeps +0.050 / +0.041 of the argmax's +0.081 / +0.071 (it switches on 18 % /
  17 % of states vs ~50 %).
- **Where (descriptive, end2 / prod, LEAF-K8):** flat across the policy's gap quintiles (q1 +0.078 / +0.090, q5
  +0.075 / +0.072: "focus on torn states" is still the wrong selector); not smaller with more of the opponent hidden
  (unrevealed 0–1: +0.072 / +0.061; 4–5: +0.092 / +0.074); larger early (turns 2–10 +0.082 / +0.090) than late
  (26–40 +0.056 / +0.049); re-weighting to the natural outcome rate lowers every IMP by ~10–20 %.
- **Predictions** (§5): R-LEAF and R-K right; IMP(LEAF-K8) ABOVE the predicted +0.02–0.06; R-TRUNC (predicted d = 4),
  R-PAIR (predicted no signature), R-PRIOR (predicted not detected / helps) and R-SH (predicted helps) WRONG.
- **Cost (measured, contended CPU):** a full playout is ~28 turns and ~60 forward rows (both sides); all-in CPU per
  row 6.1 ms (end2) / 3.5 ms (prod); the critic's one-ply successors 0.66 / 0.44 CPU-s per decision.

## 7. Phase 2 and 2b results (HONEST worlds; `result_det.md` / `.json`, `result_belief.md` / `.json`)

**In plain language.** When the rollouts may NOT peek at the opponent's real team — they play out in 8 guessed
teams that agree with everything our side has seen — the rollout search's whole advantage disappears: it improves
on the network's own pick by 0–2 points of win probability per decision, about what the critic alone does, and
nothing is clearly better than the critic. Guessing the opponent's Pokémon from the network's own belief instead of
Smogon usage does not help. The oracle's 7–8 points were almost all knowledge of the opponent's real moves and
sets: even when (nearly) every opponent Pokémon has been seen, honest rollouts recover only about a quarter of it.
Shorter rollouts do better than long ones in guessed worlds (the further you play a wrong world, the more it is the
wrong world that decides).

IMP over the policy's pick against the TRUE world's 24 truth seeds; 8 worlds per state (Amendment 3); [95 %].

| scorer (ARGMAX) | end2 honest (Smogon worlds) | end2 oracle, same states | prod honest (Smogon) | prod honest (BELIEF worlds, 2b) | prod oracle |
|---|---|---|---|---|---|
| CRITIC (one-ply, in the worlds) | **+0.013 [+0.001, +0.026]** | +0.027 | +0.002 [−0.010, +0.014] | +0.004 [−0.008, +0.016] | +0.012 |
| LEAF-K1 | −0.001 [−0.011, +0.009] | +0.034 | −0.002 [−0.010, +0.006] | +0.002 [−0.007, +0.010] | +0.027 |
| LEAF-K4 | +0.004 [−0.009, +0.019] | +0.067 | −0.005 [−0.016, +0.008] | +0.008 [−0.005, +0.020] | +0.058 |
| LEAF-K8 | +0.007 [−0.008, +0.022] | +0.079 | +0.004 [−0.010, +0.019] | +0.001 [−0.013, +0.016] | +0.070 |
| TRUNC-4-K1 | +0.005 [−0.011, +0.020] | +0.026 | −0.010 [−0.026, +0.005] | +0.003 [−0.012, +0.017] | +0.027 |
| **TRUNC-4-K4** | **+0.017 [+0.002, +0.031]** | +0.052 | +0.008 [−0.008, +0.023] | +0.008 [−0.006, +0.023] | +0.036 |
| MIX-K8 | +0.014 [−0.001, +0.029] | +0.087 | −0.001 [−0.016, +0.015] | −0.001 [−0.014, +0.014] | +0.075 |
| GUMBEL∘LEAF-K8 (c 0.1) | +0.013 [+0.001, +0.025] | — | −0.001 [−0.011, +0.010] | +0.002 [−0.008, +0.012] | +0.040 |

| rule | end2 | prod | verdict |
|---|---|---|---|
| **R-HONEST** (det LEAF-K8 − det CRITIC) | −0.006 [−0.022, +0.010] | +0.002 [−0.012, +0.016] | **NOT DETECTED** (both) |
| **R-INFO** (oracle − det LEAF-K8) | +0.072 [+0.057, +0.087] | +0.067 [+0.053, +0.080] | **INFO-COSTS** (both) |
| retained share (det / oracle, LEAF-K8) | 0.09 [−0.10, +0.26] | 0.05 [−0.17, +0.23] | descriptive |
| det TRUNC-4-K8 − det LEAF-K8 | +0.016 [+0.001, +0.031] | −0.005 [−0.019, +0.008] | descriptive |
| det CRITIC − oracle CRITIC (one turn of TRUE reveals in the oracle successor) | −0.014 [−0.025, −0.004] | −0.011 [−0.020, −0.000] | descriptive |
| **R-HONEST-B** (belief worlds: LEAF-K8 − CRITIC) | −0.006 [−0.020, +0.009] | −0.003 [−0.018, +0.012] | **NOT DETECTED** (both) |
| **R-BELIEF** (belief − Smogon worlds, LEAF-K8) | +0.001 [−0.014, +0.015] | −0.003 [−0.016, +0.009] | **NOT DETECTED** (both) |

- **Where the oracle value lives:** by unrevealed opponent slots, honest LEAF-K8 vs oracle: 0–1 unrevealed +0.019 /
  +0.072 (end2), +0.014 / +0.062 (prod); 4–5 unrevealed +0.008 / +0.082, −0.007 / +0.072. Even with the species
  known, the honest leaf keeps ~¼: the hidden MOVES (≈ 10 move slots re-drawn per world) carry most of the oracle's
  edge. A Smogon world puts a true hidden species in a never-seen slot 22 % of the time and the true move in a
  re-drawn move slot 16 % of the time (`world_accuracy_read.json`); the network's own species belief puts 47 % of its
  mass on true hidden species (`belief_accuracy_prod.json`) yet buys nothing in IMP (R-BELIEF), consistent with the
  moves, not the species, being what matters.
- **Validity:** the driver reproduces Phase 1's outcomes exactly when handed the TRUE team (12 prod states, 93
  actions, 0 mismatches, `det_identity_check_prod.json`); 21–22 states per checkpoint lacked 8 gate-passed worlds and
  are excluded (3.5 %). World building: ≈ 2.6 tries per world.
- **Predictions:** R-HONEST predicted LEAF-AHEAD at half the oracle: WRONG (≈ 0); R-INFO INFO-COSTS: right; R-BELIEF
  predicted HELPS: WRONG on both. End2's best honest scorer is TRUNC-4-K8, +0.020 [+0.007, +0.035] under either
  sampler (descriptive).

## 8. Phase 2: DETERMINIZED worlds (registered by the design doc's §7.10 Amendment 2)

**In plain language.** Phase 1's rollouts cheat: they play from the true battle, so they know the opponent's
whole team. Phase 2 plays the same rollouts in worlds a real player could believe in: for each decision, 16
guesses of the opponent's hidden Pokémon and moves, drawn from Smogon usage statistics and thrown away unless the
guess would have produced EXACTLY what our side saw so far. The referee is unchanged (the true world). The gap
between Phase 1 and Phase 2 is the price of not knowing the opponent's team (part of it; see the leaks).

| item | value |
|---|---|
| worlds | 16 built per state, the FIRST 8 played (Amendment 3); never-seen opponent slots sampled whole (species ∝ Smogon teammate co-occurrence with the revealed species, else usage; moves / item / ability / spread ∝ that species' usage; a Hidden Power sets its IVs), never-seen move slots of a seen mon re-drawn; only moves / items the pin's engine runs (`determinize.BANNED_*`, from `scan_move_probe`) |
| gate | the world's log, cut at the root, replayed at the pin (`core_events --trackers --obs`): the searched side's obs rows at every decision up to the root must be byte-equal to the true log's; ≤ 30 tries per world; a state enters only with all 16 worlds |
| leaks (counted) | a seen mon's item, ability, nature, EVs, IVs; an ambiguous unseen move choice kept by name; the opponent's root-turn choice when the log holds it (~1 %); the true pre-root dice (a world must match the true team's construction draws) |
| playouts | world j, dice seed j (Phase 1's leaf seed j), all actions to the end, CP greedy both sides; the opponent's open root decision answered greedily in the world |
| honest critic | one-ply successors in worlds 0–7 on seeds 40–47 |
| referee | Phase 1's TRUE-world truth, seeds 16–39 |
| rules | **R-HONEST (primary of Phase 2)**: IMP(det LEAF-K8) − IMP(det CRITIC), as R-LEAF. **R-INFO**: IMP(oracle LEAF-K8) − IMP(det LEAF-K8): INFO-COSTS / INFO-FREE (inside ±0.01) / NOT DETECTED. Retained share and strata by unrevealed slots, descriptive |
| predictions | R-HONEST LEAF-AHEAD at about half the oracle's improvement; R-INFO INFO-COSTS; the gap largest with ≥ 2 unrevealed slots |

## 9. Files

| file | what |
|---|---|
| `run_at_pin.sh` | runs a script here against a checkpoint's own pin checkout: CPU only, nice 19, mem cap, timeout |
| `common.py` | the checkpoints, the seed plan, the clock decode, durable rows |
| `probe_load.py` | load check + forward cost (no outcome) |
| `make_states.py` | the pool, gate ①, the root forward, the quintiles, the sample → archive `<cp>/bank/`, `<cp>/pool.npz`; here `states_<cp>.json` |
| `run_playouts.py` | the driver: all actions × 40 seeds to the end with the value trace, plus the critic's one-ply successors → archive `<cp>/rows/shard_*.jsonl` |
| `states_end2.json`, `states_prod.json` | the registered samples (edges, counts, ids, the per-(cycle, opponent) capture rates) |
| `shards.sh` | W shards of one driver in one capped job (`--det` = Phase 2's) |
| `score.py` | Phase 1's read: every scorer × rule, strata, costs, pairing, verdicts → `result.json`, `result.md` |
| `determinize.py`, `det_worlds.py`, `det_debug.py` | Phase 2's sampler + gate, the per-state world builder (archive `<cp>/det/worlds_*.jsonl`), a debug aid |
| `run_det_playouts.py`, `score_det.py` | Phase 2's lockstep driver (archive `<cp>/det/rows_*.jsonl`) and read → `result_det.json`, `result_det.md` |
| `phase2.sh` | one checkpoint's worlds then playouts in one capped job (`--belief` = Phase 2b into `<cp>/det_b/`) |
| `score_belief.py` | Phase 2b's read (belief vs Smogon vs oracle on common states) → `result_belief.json`, `result_belief.md` |
| `det_identity_check.py` | the Phase 2 driver's teeth: the TRUE team as every world reproduces Phase 1 → `det_identity_check_prod.json` |
| `target_scale_read.py`, `world_accuracy_read.py`, `belief_accuracy_read.py` | POST-HOC / descriptive reads (the Gumbel σ scale; how often a world guesses a hidden species / move; the network's species belief) → their `.json` |
| `move_belief.py`, `move_belief_probe.py` | Phase 2c: the network's per-mon move presence at a root (the X5 op roster's `move_w`, captured from one forward); a debug print beside the true sets |
| `move_accuracy_read.py` | DESCRIPTIVE (no outcome): per sampler (A / B / C), how often a world holds the true hidden moves (slot and set recall) and species → `move_accuracy_read.json` |
| `make_qstates.py`, `run_qlabels.py` | Phase 3's training states (every cycle minus the evaluation battles → archive `<cp>/qbank/`, here `qstates_<cp>.json`) and their oracle labels (K = 2 per action, states stepped in lockstep → archive `<cp>/qrows/`) |
| `qfeatures.py`, `fit_qhead.py` | Phase 3's frozen-checkpoint root features (archive `<cp>/qfeat_<bank>.npz`) and the Q head's fit + read → `result_qhead.json`, `result_qhead.md` |
| `qhead_noise_read.py` | POST-HOC (training states only): the labels' noise share and simple-predictor R² → `qhead_noise_read.json` |
| `score_moves.py` | Phase 2c's read (C / CT / B / A / oracle worlds on common states) → `result_moves.json`, `result_moves.md` |

`phase2.sh` also takes `--moves` (Phase 2c → `<cp>/det_c/`) and `--truemoves` (2c-T → `<cp>/det_ct/`), and
`--worlds-only` / `--playouts-only`. Bulk: `~/gen3ai_archive/value_disc_pilot/`.

## 10. Phase 2c: worlds from the network's own MOVE belief (registered by the design doc's §7.10 Amendment 6)

**In plain language.** Phase 2 found that honest rollouts lose the oracle's edge because they guess the opponent's
hidden MOVES wrong. The network has its own belief about those moves (it is trained to predict them). Phase 2c
draws each world's hidden moves from that belief instead of from Smogon usage, and a ceiling arm (2c-T) gives the
worlds the opponent's TRUE moves for every Pokémon already seen, to price how much a perfect move belief could buy.
Before any playout, a read that uses no outcome already says the network's belief is only slightly sharper than
Smogon's (half the hidden moves right instead of 45 %).

| item | value |
|---|---|
| sampler C (`det_c`) | species of a never-seen slot ∝ the network's X5 species presence (2b's); a seen mon's unseen move slots, and a never-seen slot holding one of the root's hypothesis species, drawn from the network's per-mon move presence `move_w` (fixed mass 4 − r over the species' legal moves minus the revealed ones; the composed posterior = Smogon prior + learned correction) as a conditional-Poisson fixed-size sample; engine-runnable moves only; ≤ 1 Hidden Power, of the kept IV type on a seen mon; a never-seen slot's item / ability / spread from Smogon; every Phase 2 leak kept |
| ceiling 2c-T (`det_ct`) | sampler C, but every SEEN mon keeps its TRUE set |
| worlds | 8 built per state (the 8 played), ≤ 60 tries each (attrition only); gate, rewrites, seeds, honest critic and referee as Phase 2 |
| rules | **R-MOVES (primary):** det-C LEAF-K8 − det-B LEAF-K8: HELPS / HURTS / NOT DETECTED. **R-HONEST-C:** det-C LEAF-K8 − det-C CRITIC: LEAF-AHEAD / CRITIC-EQUIVALENT / NOT DETECTED. **R-MOVES-CEILING:** det-CT LEAF-K8 − det-C LEAF-K8: HELPS / HURTS / NOT DETECTED. Descriptive: CRITIC / TRUNC-4-K4 / TRUNC-4-K8 / LEAF-K8 in every world kind; the closed share of the oracle − honest gap |
| disclosure (no outcome) | `move_accuracy_read.json` (558 / 568 common states): re-drawn slot = true move 18.7 % / 17.2 % (Smogon 16.6 % / 16.0 %); SET recall of the true unseen moves 51 % / 48 % (Smogon 45 % / 45 %); the network's own expected share 47 % / 43 % |
| predictions | R-MOVES NOT DETECTED; R-HONEST-C NOT DETECTED; R-MOVES-CEILING HELPS by +0.01 to +0.03, det-CT still clearly below the oracle |

## 11. Phase 3: an OFFLINE per-action value head on oracle labels (registered by Amendment 7)

**In plain language.** The oracle rollouts know the opponent's team, so they cannot CHOOSE moves honestly. But they
can TEACH: a model that sees only what we see, trained to predict how each action played out from the true
position, learns the average over every team the opponent could have had given what we saw, which is exactly the
honest value. Phase 3 trains a small per-action head on top of the frozen end-state network (no fine-tuning) on
oracle outcomes of ~9,600 OTHER positions, then plays its favourite action on the 600 test positions with no search
at all, and scores it with the same held-out referee.

| item | value |
|---|---|
| why unbiased | E[oracle playout of a | what we observed] = the information-set value of a under the data's own posterior, provided the playout continuation acts on its own observations: verified at the pin (each side folds its own stream with its own team; no oracle-reveal input on the playout path) |
| training | `make_qstates.py`: all 7 cycles of `rb_es_end2_s2001` minus the 417 evaluation battles = 964 battles, gate ① 30,139 / 30,139, 20,719 eligible, ≤ 10 per battle = **9,629 states**; labels K = 2 oracle playouts per legal action (`run_qlabels.py`, seeds disjoint from every evaluation seed) |
| features | the frozen end2 checkpoint at its pin: `value_pooled`, the pointer head's per-action inputs and hidden layer (reproduces the logits ≤ 1e-4), logit, log π, V |
| head | family projection + context projections → ReLU → 256 → ReLU → 1; per-state CENTRED squared error; AdamW 1e-3, wd 1e-4; early stopping on a 15 % battle split of the training states; 5-seed ensemble; argmax, ties to the policy's order |
| evaluation | the 600 registered states, truth seeds 16–39; no rollout or world at decision time |
| rules | **R-QHEAD (primary):** QHEAD − policy: HELPS / HURTS / NOT DETECTED. **R-QHEAD-CRITIC:** QHEAD − honest one-ply CRITIC (Phase 2, common states): AHEAD / BEHIND / EQUIVALENT (±0.01) / NOT DETECTED. Descriptive: vs honest TRUNC-4-K4, oracle CRITIC, the ride-along A head; the label-size curve; validation R²; within-state Spearman; share moved off top1 |
| predictions | R-QHEAD NOT DETECTED or HELPS ≤ +0.015; R-QHEAD-CRITIC NOT DETECTED; validation R² < 0.05; the A head ≈ 0 |
| vs the old null | `crn_label_refit_2026-10-06/` refit a STATE-value readout on one-ply successor features (a search), older architecture, ~800 turns, sibling accuracy; this is a per-ACTION head at the root, end-state checkpoint, ~9,600 states, decision currency. Shared limit: a frozen body |

## 12. Phase 3 results (read 2026-10-11 20:42; `result_qhead.md` / `.json`, `qhead_noise_read.json`)

**In plain language.** The per-action value head learned nothing it could use. Trained on the oracle outcomes of
9,629 other positions, it predicts which action is better no better than chance on held-out training battles
(validation R² +0.002), and acting on it does not beat the network's own pick (−0.004 per decision, not clear of
0) and is WORSE than the honest one-ply critic search (−0.018, clear). The reason is visible in the labels: with two
playouts per action, 83 % of the spread between sibling actions' labels is dice noise, and of the rest most is what
the hidden team decides, so the part a model that sees only our observations can learn is tiny; the small head
over-fits it within two epochs. A surprise on the side: the network's OWN on-policy per-action readout (the
ride-along A head, which nobody trained for this) improves the pick by +0.010 [+0.001, +0.020] at zero extra cost,
close to the honest critic search (descriptive, not registered).

Training 9,629 states / 964 battles (validation 145 battles); evaluation the 600 registered states (578 with Phase 2's
honest rows); IMP on the win scale [95 %, battle-clustered]; "moved" = the share of states whose pick differs from the
policy's.

| scorer (argmax) | rollouts / worlds at decision time | IMP [95 %] | moved |
|---|---|---|---|
| **QHEAD** (5-seed ensemble, all fit battles) | none | **−0.004 [−0.016, +0.009]** | 0.40 |
| QHEAD at 25 % / 50 % of the fit battles | none | −0.009 / −0.000 | 0.46 / 0.45 |
| ride-along A head (`adv_mean`, the network's own) | none | **+0.010 [+0.001, +0.020]** | 0.20 |
| honest CRITIC (Phase 2, Smogon worlds) | 8 worlds × one ply | +0.013 [+0.000, +0.025] | 0.44 |
| honest TRUNC-4-K4 (Phase 2) | 4 worlds × 4 turns | +0.017 [+0.001, +0.031] | 0.55 |
| oracle CRITIC / oracle LEAF-K8 (Phase 1) | true world | +0.026 / +0.081 | 0.47 / 0.51 |

| rule | contrast | Δ [95 %] | verdict |
|---|---|---|---|
| **R-QHEAD (primary)** | QHEAD − policy | −0.004 [−0.016, +0.009] | **NOT DETECTED** |
| **R-QHEAD-CRITIC** | QHEAD − honest CRITIC (578 common) | −0.018 [−0.032, −0.004] | **BEHIND** |
| descriptive | QHEAD − honest TRUNC-4-K4 / − oracle CRITIC / − A head | −0.021 / −0.031 / −0.015 (all clear of 0) | — |

- **Why (POST-HOC, training states only, `qhead_noise_read.json`):** the centred labels' variance is 0.066 per action,
  of which the K = 2 dice noise is **83 %** (an R² ceiling of 0.17 on these labels, and that remaining 17 % includes
  the advantage the hidden team decides). The head's validation R² is +0.002 (25 % / 50 %: −0.004 / −0.001; best
  epoch 1–2, then over-fitting); one slope on the policy logit alone reaches +0.005, on the A head +0.005. Within a
  state, the head's ranking correlates with the truth at Spearman ρ 0.10 against the policy logits' 0.14, the honest
  critic's 0.15 and the oracle critic's 0.20.
- **Predictions** (Amendment 7): R-QHEAD NOT DETECTED right; R-QHEAD-CRITIC predicted NOT DETECTED, read BEHIND;
  validation R² < 0.05 right; the A head ≈ 0 WRONG (+0.010, clear of 0, descriptive).
- **Against the old null:** the CRN refit (a state-value readout on one-ply successors, older architecture) and this
  per-action head at the root agree: a FROZEN body with a readout refit on a few oracle labels per action does not
  learn to separate siblings. What the two leave open is the label budget (labels with far less noise per action,
  or many more states) and a TRAINED body (the GPU step, deferred); the noise read says the part of these labels a
  simple observation-side predictor explains is ~0.5 % of their variance, so the signal a head must find is small
  against the noise at this budget (how many more labels would suffice is UNVERIFIED).
- **Validity:** the training states share no battle with the evaluation states (checked); every training label's
  action set equals its state's mask (checked); the features' pointer hidden layer reproduces the policy's scores up
  to the per-row normalisation (max |Δ| 1e-6). Disclosure: the "logit" scalar in the features is the distribution's
  normalised log-probability, so the registered (logit, log π) pair is the same number twice.
- **Cost:** a root forward the policy already makes plus a small MLP (≈ 0 extra), the cheapest scorer here; the
  labels cost ≈ 14 worker-hours for 9,629 states (6 single-thread workers on 3 contended physical cores for 2.4 h,
  states in lockstep groups of 16).
