# The value of information by STAGE of the game, and the belief-quality dose-response (2026-10-12)

**Status: (A) the per-stage descriptive read and (C) the belief-quality dose-response READ (post hoc, descriptive, on the
original study's EXISTING rows; no new playout, nothing blinded because nothing here is a verdict). (B) the REGISTRATION
that follows from them is `designs/endstate/design_hidden_info_binding.md` §6.9 and §12.13 (Amendment 2), written before
any new arm is run.** The blind replica's code, results and archive were not read (owner's rule).

Owner's questions (2026-10-12): *where does improved determinization make the most difference, and where does search
already add the most?* — *is the value of perfect information concentrated EARLY (much hidden, maybe low stakes) or LATE
(little hidden, decisive stakes)?* — *how far are we from determinized worlds improving ALL of search?*

## 1. In plain language

**The value of perfect information is concentrated EARLY and in positions with plenty of material left, and it is
concentrated there because the truth would CHANGE the decision far more often, not because more is at stake when it does.**

- **How big, by stage.** Oracle search (it plays out every action in the TRUE battle) beats the policy's own pick by about
  7 points of win probability per decision *at every stage* (+0.081, +0.065, +0.099, +0.068, +0.066 at 1–2, 3, 4, 5, 6
  opponent Pokémon seen: flat, as the pilot hinted). What changes is how much of that an HONEST search (worlds guessed
  from Smogon usage, kept only if they reproduce everything we saw) recovers: −0.004 / −0.016 / +0.023 / +0.012 / +0.031.
  So the value of perfect information, G = oracle − honest, falls with the reveal count:
  **+0.086 / +0.081 / +0.076 / +0.056 / +0.035** (95 % intervals ±0.02–0.03; the paired contrast k ≤ 3 minus k ≥ 5
  is +0.038 [+0.017, +0.059]; on production alone k = 6 is +0.012 [−0.002, +0.030]: nothing left to find).
- **Frequency × stakes, so the mechanism is visible.** G is exactly (how often the oracle's pick differs from the honest
  pick, c) × (the true-value gain per decision that does differ, s). **s is the same at every stage (0.104–0.115; the
  early − late difference is +0.001 [−0.037, +0.038])**; **c falls from 0.78 (k ≤ 2) to 0.30 (k = 6)** (early − late
  +0.352 [+0.291, +0.413]). Late in the game honest search already agrees with the oracle in 70 % of decisions; early it
  disagrees in 78 %. It is a FREQUENCY effect: when the choice flips, it flips by the same amount at every stage.
- **The stage that matters is the board, not the reveal count.** Regressing the per-state gap on k, turn and the
  material left (unfainted mons, ours + theirs), only **material** keeps its slope once k is in the model (k: −0.002
  [−0.017, +0.012]; material: +0.022 [+0.010, +0.035] per SD; pooled). At equal material (≥ 8 alive) k ≤ 3 and k ≥ 5 have the
  same gap (+0.013 [−0.019, +0.047]). The gap is large with ≥ 8 mons alive (+0.073 / +0.086 / +0.087 at 8–9 / 10–11 / 12)
  and small with ≤ 7 (+0.031 [+0.018, +0.045]); by turn: +0.091 (≤ 7), +0.066, +0.074, +0.039 (≥ 22).
- **WHICH hidden fact, by stage** (the CT′ world kind: the network's belief worlds plus the seen Pokémon's TRUE sets).
  Early (k ≤ 3, turn ≤ 7, 12 alive) the seen Pokémon's moves explain NOTHING of the headroom (share 0.00–0.09, intervals
  spanning 0): what is missing is the never-seen Pokémon (their identity and sets). In the middle (k = 4: share **0.55
  [0.27, 0.81]**; turn 15–21: 0.54 [0.25, 0.76]; alive 8–9: 0.43 [0.14, 0.64]) the seen Pokémon's unseen moves carry about
  half. At k = 6 nothing is unseen, so the whole remaining headroom (+0.021) IS the seen moves (O − CT′ = 0.000 on all 232
  states: a built-in identity check of the world builder).
- **Where honest search already earns its keep.** Only late and in material-poor positions: at k = 6 the honest 4-turn
  search beats the one-ply critic search by +0.026 [+0.004, +0.051] and the policy by +0.041 [+0.018, +0.066] (pooled; per
  checkpoint the lower bounds straddle 0: end2 +0.022 [−0.005, +0.052], prod +0.031 [−0.001, +0.075]); with ≤ 7 mons alive
  it beats the policy (leaf +0.022 [+0.007, +0.038], 4-turn +0.020 [+0.006, +0.036]) but not the critic search (+0.009
  / +0.007, intervals spanning 0). Early (k ≤ 3) honest rollouts are at or below the policy and the critic (not clearly
  below). That is today's FRONTIER: a late-game, pooled-only edge.
- **How far are we from determinized worlds helping ALL of search (the belief-quality dose-response, §4).** Give the
  TRUE world weight p inside the search's average and the honest worlds 1 − p. The decision improvement climbs steeply and
  flattens: pooled **+0.009 (p = 0) → +0.034 (0.1) → +0.062 (0.35) → +0.069 (0.5) → +0.076 (1.0)**; half of the whole rise
  is in by p ≈ **0.22 [0.17, 0.27]** (0.23–0.31 at k ≤ 3, turn ≤ 7, 12 alive; 0.16–0.25 in the middle; 0.04 at k = 6).
  Measured in the same units, our
  best honest sampler (the network's own species + move belief) sits at p ≈ **0.01 [0, 0.03]**, and even a world that knows
  every seen Pokémon's TRUE set (CT′) at p ≈ 0.045 [0.03, 0.19]: today's beliefs are an order of magnitude short of the
  region where search pays. The break-even for the rollout leaf to clearly beat the critic-only search is p ≈ 0.05 (point)
  and 0.05–0.35 (95 % lower bound, larger early). A confidently wrong single world does not help at any weight: the
  true world adds +0.012 [+0.009, +0.014] (p = 0.1) and +0.026 [+0.022, +0.030] (p = 0.35) over a random honest world at
  the same weight (literal K = 5 design, one playout per world).

**The priority read for determinization work (descriptive; the registered arms of §12.13 test it).**
1. **Early (k ≤ 3, turn ≤ 7, material ≥ 10): predict the UNSEEN Pokémon (species and sets).** This is where the value
   of perfect information is largest (+0.086, honest search is below the policy) AND where the seen mons' moves explain
   nothing. Sampled worlds matter most here, and they must be better than Smogon usage (R-BELIEF was null, so the
   species belief that exists today is not enough); the dose-response says the early curve is the least concave (half
   the rise needs p ≈ 0.25–0.31), so only a large improvement shows.
2. **Middle (k = 4, turns 15–21, 8–9 alive): infer the SEEN Pokémon's moves.** Half of the headroom is the seen mons'
   unseen moves (share ≈ 0.55), the dose-response is steep (p-half 0.16), and the break-even p is small (0.05).
3. **Late (k = 6, ≤ 7 alive): little is left (G +0.035; prod +0.012), the policy and critic are already close.** Here
   better determinization buys least; this is where honest search is already ahead of the critic, so a first
   search-label factory would start here (design_value_discrimination §13.10).
4. **Throughout:** the owner's "five competing worlds with variable mass on the correct one" confirms the lever is
   large only if beliefs can put real mass on the whole true world; no per-fact hit rate we have measured does (§4.3).

## 2. What this read may and may not do

It is POST HOC and DESCRIPTIVE: read after the original study's rows (and its descriptive reads D1–D9, `hidden_info_binding_2026-10-11/`)
existed, on those same 1,200 states. It may not be quoted as a verdict, and it may not steer the registered rules of
the amendment. What it DID fix in the amendment (and nothing else): the bucket edges (quartiles of the registered
states' turn and material, no outcome), the k-merging rule (n < 80 per checkpoint, no outcome), and the SIZES of the
state sample (from the per-state standard deviations of the contrasts, `by_reveal_count.json` `power`). Every
threshold in the amendment is inherited from an existing rule or derived from a measured interval width, and the
amendment's predictions are labelled as informed by this read.

## 3. Method

States: the pilot's 600 registered states per checkpoint (`rb_es_end2_s2001`, `rb_es_prod_s2001`, 15M, own eval traces,
stratified by the policy's top-2 gap = a proportional sample of the eligible pool, loss-enriched), the 591 / 595 on which
the oracle and all four v2 honest world kinds played (the pilot's `result_moves` common set; the means reproduce its
table to 5e-6, enforced by the script). Referee: the pilot's 24 truth seeds (16–39), which no scorer reads.

| axis | definition (all from the SEARCHED side's own view; no outcome) |
|---|---|
| **k** | the number of distinct opponent species seen switch or be dragged in up to the root (= 6 − the pilot's `unrevealed`; equal to `root_facts` `n_seen` on every state). Bins: **1–2** (k = 1 has 62 / 67 registered states, below 80 per checkpoint, so it merges up into k = 2: 153 / 180), 3, 4, 5, 6 |
| **turn** | the state's turn (obs clock, gate-checked by the pilot); quartile buckets of the 1,200 registered states: ≤ 7, 8–14, 15–21, ≥ 22 |
| **material** | unfainted mons, ours + theirs (2–12), `stage_facts.py` (the protocol replay; 0 refused, 0 unreadable HP, fainted = HP 0 on every state); quartile buckets ≤ 7, 8–9, 10–11, 12. (The remaining-HP fraction is in `stage_facts_*.json`; it correlates 0.95 with the count and is not used.) |

Correlations: k–turn 0.66, k–material −0.71, turn–material −0.56 (the axes are NOT independent; the separation above
rests on within-cell variation and a regression, and the registered sample balances them).

IMP is the pilot's decision improvement T(pick) − T(policy top1) on the win scale. World kinds: O oracle (Phase 1, true
world, leaf seeds 0–7), A′ honest Smogon-prior worlds (v2 builder, 8 worlds × one playout), B′ + network species belief,
C′ + network move belief, CT′ = C′ + the seen mons' true sets. All v2 honest worlds keep a seen mon's item, ability and
spread TRUE (the pilot's leak, `design_hidden_info_binding.md` §2.2), so G here is the pilot's M + U gap, not the fully
honest gap F. `by_reveal_count.py` computes the cells (battle-clustered bootstrap on the pilot's draws, raw is primary;
the outcome-reweighted G is in the JSON); `belief_dose.py` the dose-response; `render_tables.py` writes `tables.md`.

## 4. The belief-quality dose-response (the owner's "five competing worlds")

**The idea.** Search averages K worlds; the TRUE world gets weight p, the honest worlds share 1 − p. The playouts do not
depend on p, so one set of per-world outcomes gives the whole curve by re-weighting; the archive banks exactly them
(Phase 1: 16 leaf seeds of the true world; `det2`: 8 honest worlds, one playout each), so NO playout was run.

| estimator | score of action a | what it is |
|---|---|---|
| **M8** | p · mean(true world, 8 seeds) + (1 − p) · mean(8 honest worlds) | the two ends are the pilot's own banked reads (p = 0 the honest A′ leaf, p = 1 the oracle leaf); the SHAPE between them is new. Scorers: leaf K8, 4-turn leaf K8, one-ply critic |
| **M5** | p · (true world, ONE playout, seed s0) + (1 − p)/4 · (4 of the 8 honest worlds, one playout each) | the literal K = 5 design, averaged over all 16 × 70 (s0, subset) draws (exact, no Monte Carlo) |
| **M5W** | the same with ONE honest world (a random draw of the belief, confidently over-weighted) in the true world's place | the placebo: M5 − M5W at the same p is the value of the world being TRUE rather than of concentrating weight |

Per cell: the curve, the break-even **p\*** (the smallest p after which the contrast stays above the 0.0005 edge, by the
point estimate and by the 95 % lower bound) where the honest-world leaf (a) beats the policy, (b) beats the critic-only
search at the same p, and the **IMP-equivalent p** of the world kinds we have (the p at which the M8 curve equals a world
kind's own measured IMP: today's belief in units of VALUE). Pooled, both checkpoints:

| | p = 0 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| leaf K8 (M8) | +0.009 | +0.034 | +0.040 | +0.062 | +0.069 | +0.078 | +0.076 |
| 4-turn leaf (M8) | +0.014 | +0.021 | +0.033 | +0.040 | +0.044 | +0.043 | +0.044 |
| one-ply critic (M8) | +0.010 | +0.010 | +0.010 | +0.011 | +0.014 | +0.018 | +0.019 |
| leaf, K = 5 once (M5) | +0.007 | +0.018 | +0.025 | +0.032 | +0.032 | +0.032 | +0.028 |
| same, a WRONG world (M5W) | +0.007 | +0.006 | +0.008 | +0.006 | +0.006 | +0.005 | +0.003 |

**4.1 The shape.** Concave: the first 10 % of truth buys 37 % of the oracle's rise, 35 % buys 79 %, 50 % buys 90 %; the
one-ply critic barely moves (+0.010 → +0.019): a critic cannot use the extra information, a rollout can. Per stage
(`tables.md`): the early and material-rich stages need the most truth for the same share (p-half 0.23–0.31 at k ≤ 3,
turn ≤ 7, 12 alive), the middle and late buckets 0.16–0.25, and the end (k = 6) only 0.04. **Why it is so concave (a mechanism, not a theorem):** the honest worlds' mean
is nearly indecisive at the top (their action values are flat), so a small true component breaks the near-ties toward
the truth; the true world's action values are only 1.1–1.4× as spread as the honest mean's (`belief_dose.json`).

**4.2 Where today's belief sits, and the distance.** In value units (p-equivalent, pooled): the network's own belief
worlds C′ **0.010 [0.000, 0.029]**; B′ (species belief) 0.000; CT′ (seen mons' true sets) 0.045 [0.027, 0.19]. The
break-even p\* of the leaf over the critic search: 0.05 pooled (point) with a 95 % lower bound of 0.05; by stage (95 %
lower bound) k 1–2 0.20, k 3 0.35, k 4 0.05, k 5 0.25, k 6 0.05; turn ≤ 7 0.20, 8–14 0.05, 15–21 0.05, ≥ 22 0.25; alive
≤ 7 0.05, 8–9 0.05, 10–11 0.20, ≥ 12 0.35. So: **early and material-rich stages need 0.20–0.35 of truth-weight for the
leaf to CLEARLY beat the critic search (20–35× today's ≈ 0.01); late stages need ≈ 0.05 (a few times today's)**.
Intervals are wide (n ≈ 200–330 per stage): read the pattern, not a digit.

**4.3 Units warning.** p is a weight on the exact true world; no real belief puts mass on a whole team, so p is NOT a
probability the network holds. Two bridges, which disagree: (i) value units (above); (ii) hit-rate units, p_eff =
(h − h₀)/(1 − h₀) from the pilot's own world accuracy (`value_disc_pilot_2026-10-11/move_accuracy_read.json`, v1
builder): species hit 0.45 vs Smogon 0.21 ⇒ 0.30–0.31; seen-mon move SET recall 0.51 / 0.48 vs 0.45 ⇒ 0.11 / 0.065;
single move slot 0.187 / 0.172 vs 0.166 / 0.160 ⇒ 0.026 / 0.015. The network's better species hit rate buys NO value
(B′ ≈ 0), so hit rates overstate what partial correctness is worth (value needs the whole world to hang together); the
registered rules therefore use value units.

**4.4 Caveats.** The honest component is A′ (it keeps a seen mon's item/ability/spread true), so p = 0 is not the fully
honest F. The true and honest components share dice seeds 0–7 (a correlated-noise detail, no bias against the referee).
Selection: many cells, no multiplicity adjustment; per-stage p\* are post hoc hints. M5W is a random draw of the belief
(over-confidence in ONE sample), not an adversarially wrong world (that is registered, §12.13.3).

## 5. Findings (standing rule 7)

1. **Cost repricing.** The rows' own CPU fields (`costs.json`, contended box): Phase 1 oracle truth **98 / 58 CPU-s per
   state** (end2 / prod), an honest arm 24 / 15 (+ world building ≈ 0.3–5), against `design_hidden_info_binding.md` §15's
   "≈ 4 CPU-h per Phase 1, 2.6 CPU-h per fair arm (prod)". At 600 states: Phase 1 16.4 / 9.7 CPU-h, a fair arm 4.0 / 2.4
   CPU-h (before the F sampler's heavier world building); Tier 1's 9 arms × 2 checkpoints ≈ 58–65 CPU-h, not 47. §15 is
   re-priced in the amendment.
2. **k = 6 identity.** O − CT′ is exactly 0 on every k = 6 state (232): with nothing unseen, the CT′ world IS the true
   world, so the builder's "true team through v2" holds at scale (beyond the 150-state A1 audit).
3. **Material, not reveal count,** carries the stage effect (§1); the registered sample balances k against material.
4. **Overall G on the 4-arm common set** equals the pilot's R-INFO (+0.070 end2, +0.065 prod); the common set excludes
   the ~1 % of states without 8 worlds in some arm.
5. **The S0 stage gate is open nowhere under a strict (per-checkpoint) rule today;** pooled, only k = 6 (4-turn leaf over critic).
6. The pilot's phrase "oracle edge similar all-seen vs unseen" holds at every stage; the VOI lives in the HONEST side.

## 6. Files

| file | what |
|---|---|
| `stage_facts.py`, `stage_facts_<cp>.json`, `run_at_pin.sh` | material per registered state, from the searched side's protocol at the checkpoint's own pin (replay only; 1.7 GB peak, 1 thread) |
| `brc_data.py` | the loader (the pilot's own scorers, read-only), the per-state bundle cache (`~/gen3ai_archive/hidden_info_by_reveal_count/`), the bootstrap helpers |
| `by_reveal_count.py` → `by_reveal_count.json` | (A): per-stage cells, trends, mechanism contrasts, power inputs |
| `belief_dose.py` → `belief_dose.json` | (C): the K-world mixture curves, p\*, p-equivalents, p-half |
| `render_tables.py` → `tables.md` | every table, per scope (pooled, end2, prod) |
| `measure_costs.py` → `costs.json` | the measured CPU cost per state of the pilot's arms |
| `belief_dose_test.py` | teeth on synthetic data: the tie rule, the M8 endpoints, M5 = M5W at p = 0, the p-star / p-equivalent helpers, the HP parser (`python -I -m pytest --noconftest -q belief_dose_test.py`) |
| `run_read.sh` | runs any pure-numpy read here under nice 19, taskset 5-7,13-15, mem_cap, a timeout |

Run: `./run_at_pin.sh <end2|prod> 1 6 30m stage_facts.py` (once each), then `./run_read.sh by_reveal_count.py`,
`./run_read.sh belief_dose.py`, `python -I render_tables.py`.
