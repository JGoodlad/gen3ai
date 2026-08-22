# Decomposing "the critic was wrong" — the win-prob error taxonomy

*Built empirically over 2026-08-21/22 (the G0 bias map → the three-axis probe → the
hidden-information floor). "Critic surprise" began as one number and ended as a five-way
decomposition, each component with its own instrument, its own size, and its own lever — or a
proof that no lever exists. The reference numbers are gen-17 @24M; the structure is durable.*

## 1. Intuitive: "the model was confident and lost" is five different sentences

A head says P(win) = 0.85 and the game is lost. That observation is compatible with ALL of:
(a) the position really was ~0.85 and the 15% tail arrived (dice); (b) the position was ~0.85
*given what was visible* and the hidden slot happened to be the one counter (a coin the head is
not allowed to see); (c) the head is systematically optimistic (a level error); (d) the head
cannot tell this position apart from genuinely-winning ones it files under the same 0.85 (a
resolution error); (e) the head is calibrated for a different opponent population than the one it
just played. Each has a different cure, and several have none. The decomposition is the act of
measuring which sentence is true, in what proportion.

## 2. The five axes, in discovery order

**Axis 1 — realized outcome vs true probability (the LUCK split).** One game outcome is ONE
Bernoulli sample of P(win); it cannot distinguish "confidently wrong" from "correctly confident,
unlucky". Instrument: tight-MC — R rollouts of the same state (fresh dice, live opponents) turn
±0.5 outcome noise into a ±0.17 (R=8) estimate. Measured: **53% [42, 65] of the 0.827 conviction
class was genuinely winning** (MC ≥ 0.75 — the dice lost the game); only ~30% read MC < 0.5. Rule
minted: never re-quote a single-outcome "surprise" statistic as critic error. Lever: none —
luck is not a defect (though tight-MC as an *evaluation* variance reducer is the AIVAT idea).

**Axis 2 — calibration vs resolution (the Murphy split).** The Brier score decomposes into
RELIABILITY (when you say 70%, does it happen 70% of the time — a mean, per bin) and RESOLUTION
(do your forecasts separate the outcomes). A base-rate forecaster is perfectly calibrated and
useless. Measured: population-mean gaps **|0.05–0.07|** (reliability tolerable) against a
within-decile TRUE spread of **0.11–0.36, 80–95% real state variance** (resolution is the
disease — the head is BLURRY, lumping 36-point-different positions into one bin). Consequence:
the meter is `sd_true_excess`, never the mean gap — a re-centred head fakes success on the wrong
meter. Lever: prioritized low-variance supervision of the rare conviction region (R1).

**Axis 3 — the population (the ECOLOGY split).** The head is BCE-trained on a ~90% self-play
mixture where P(win) ≈ 0.5, so its mean bias FLIPS SIGN by opponent: **−0.065 vs bots
(under-predicts), +0.058 vs pool (over-predicts)**. "The critic is optimistic by X" is
meaningless without naming the population — the sign depends on it. Lever: opponent diversity
(the flywheel, seen from the critic's side) and population-stratified reads everywhere.

**Axis 4 — learnable vs irreducible (the FLOOR split).** Tight-MC labels are OMNISCIENT (the
reconstruction record knows the hidden team); the head sees only our information set. So the
measured blur = learnable blur + the irreducible variance of the opponent's hidden half.
Instrument: pool-consistent DETERMINIZATION of the never-revealed slots (swap them, verify the
prefix replays byte-identically — 1,150/1,150 — roll each world out; the across-world spread of
P(win) is the floor at that state). Measured: **floor sd 0.151 [0.119, 0.186] in deciles 7–9 =
39% [24, 87] of the meter's variance; ~⅓ of the conviction region.** The canonical state:
predicted 0.879, ONE hidden slot — Salamence ⇒ 0.125, Gyarados/Skarmory/Gengar/Charizard ⇒
1.000, Vaporeon ⇒ 0.000. A coin, not an error. Structure: **concentrated, not a fog** — 49% of
states carry ~zero floor, the top 10% carry half; FLAT in hidden-slot count and game turn, so the
governing quantity is *"does the unknown decide this position"*, not "how much is unknown".
Lever for the floor states: a better opponent-TEAM BELIEF (knowing the coin) — a better value
head is provably not the lever there. Rules: effect sizes on the EXCESS over the floor; flat
verdicts evaluated floor-subtracted; arm-vs-control differences primary (the floor is a
population property and cancels).

**Axis 5 — the model's own confessed uncertainty (the EPISTEMIC layer).** All four axes above
are measured from outside. The evidential Beta head (`cf_evidential`, dormant) is the inside
view: Beta(α, β) over P(win), where WIDTH is the head saying "I don't know" — distinct from
"it's a coin flip" (a sharp 0.5). Its pre-registered meter: predicted width should CORRELATE
with measured `sd_true_excess` per stratum (`width_vs_blur_spearman`). It cannot remove blur
(same trunk read); it can localize it — which the label factory's priority sampler and the
awareness stack consume. Note the aleatoric side itself splits in kind: dice-aleatoric resolves
through PLAY (future randomness), hidden-info-aleatoric resolves through REVELATION (present
unknown state) — same irreducibility to the head, different dynamics over a game.

## 3. The assembled tree (conviction region, gen-17 @24M)

```
"predicted 0.85, lost"
├── ~53% LUCK — position was winning, dice lost it        [tight-MC]  lever: none
└── ~47% the head was actually wrong, split as:
    ├── level (mean) error: small, |0.05–0.07|,           [strata]    lever: ecology
    │     SIGN FLIPS by population
    └── RESOLUTION error (blur), sd 0.11–0.36:            [sd_true_excess]
        ├── ~⅓ IRREDUCIBLE — hidden-team coins,           [determinization]
        │     concentrated in ~10–20% of states                       lever: TEAM BELIEF only
        └── ~⅔ LEARNABLE — rare-region under-supervision  [R1's target] lever: tight-MC labels
```

Sibling decomposition, not to confuse with this one: the THREE-AXIS probe decomposed the
*bootstrap target's variance* (opponent action 59.7% ≫ dice 26.5% > our action 10.0%,
behavior-weighted) — that is "what makes the LABEL noisy", where this note is "what makes the
PREDICTION wrong". They meet at the estimator design (R rollouts kill the dice term; the
opponent term is why R matters).

## 4. What this buys operationally

- Any "critic surprise" claim must state its axis: single-outcome numbers are luck-confounded,
  pooled numbers are population-confounded, raw blur numbers are floor-confounded. The honest
  quote names all three: population, outcome-stratum, floor-subtracted.
- The decomposition ORDERS the levers: R1 attacks the largest learnable share; the ecology is
  the flywheel's side effect; the floor states are banked as the first value-denominated case
  for the hidden-team belief line (a Value-of-Information number by another route).
- The method generalizes: any scalar disagreement between a prediction and an outcome, anywhere
  in this project, admits the same ladder — sample the outcome (luck), stratify the population
  (ecology), bin the predictions (resolution), determinize the hidden state (floor).

**The question you can answer after this note:** *the R1 arm runs and `sd_true_excess` in the
conviction region falls 0.325 → 0.26 — success?* Floor-subtract first: learnable excess was
√(0.325² − 0.199²) ≈ 0.257 → √(0.26² − 0.199²) ≈ 0.167 — a ~35% cut of the learnable variance,
reading as only ~20% raw. Then check the arm-vs-control difference at matched step (the floor
cancels), the calibration strata for collateral, and the budget-matched control. Raw-meter
readings understate the arm by construction — that is exactly why the amendment exists.

Related: [`credit_assignment_and_value_errors.md`](credit_assignment_and_value_errors.md) (the
four critic-failure causes this taxonomy refines), [`imperfect_information_and_equilibria.md`](imperfect_information_and_equilibria.md)
(the PBS irreducibility axis 4 measures), [`marginalization_and_uncertainty.md`](marginalization_and_uncertainty.md)
(determinization vs expectation — axis 4's instrument is PIMC pointed at a meter).
