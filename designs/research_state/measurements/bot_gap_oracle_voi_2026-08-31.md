# SI-2 — the generalist's bot gap: attribution + the oracle-opponent VoI (2026-08-31)

**Second VoI gate of the opponent-SKILL conditioning candidate (ledger 8c1c2e8).** The rare
advantage: the eval bots are scripted, so the recorded bot action at every decision IS the true
oracle opponent model (deterministic bots exactly; the two stallers up to one Protect coin;
`random` is exactly uniform-over-legal). That makes "what is knowing-it's-a-bot worth?"
measurable offline, with no retraining.

**Subject:** `ai_v9_29_rev1_0823` (the generalist, rev-1 final, 24M steps, current arch
`gen3_critic_route_wave_v1`). Gap measured over its last 3 eval steps (20M/22M/24M): **loss rate
vs bots 7.4%** (199/2700; 8.3% excluding `random`), per-bot finals 0.88–1.00 — the owner's
"~90%" framing confirmed. Traces cover 85.9% of those losses (171 loss battles with
reconstruction siblings).

## Headline: the three-way attribution of the 7.4pp gap

| bucket | evidence | share |
|---|---|---|
| **DICE** | falsify (re-rolled real turns): 23.2% CI [18.6, 28.0] of crater-decision mass is bad-tail luck; **7.0% CI [3.5, 11.1] of losses are pure-luck battles** (every falsified anchor LUCK) | ~0.5–1.7pp of the gap |
| **CONDITIONING** (opponent-model de-mixing) | decision-level oracle VoI at loss craters: **+0.082 wp CI [0.063, 0.101]** — ~8% of losses ≈ **~0.6pp of the gap** | small but real |
| **CREDIT / valuation** | proven MISTAKE 11.0% CI [7.6, 14.9] + the 60.7% unattributed residual, and the decisive tell: **one-ply search WITHOUT the oracle already recovers 0.196 of the 0.278 crater improvement** — the model's own value function finds better lines against its own marginal opponent model; what it lacks is not opponent identity | the bulk |

## Measurement 1 — falsify attribution of the recorded losses

171 loss battles × worst-2 anchors (342 → 336 scored decisions), 32 re-roll seeds, 2 alts, rust
driver. Decision count shares (cluster bootstrap over battles; |δ|-weighted shares within 1pp of
these):

- proven MISTAKE 0.110 [0.076, 0.149] · MIXED 0.051 [0.030, 0.074]
- LUCK 0.232 [0.186, 0.280]
- NEUTRAL (unattributed) 0.607 [0.553, 0.662]

Battle level: 12/171 pure-luck, 46/171 contain a proven-or-mixed mistake, 70/171 neutral-only.
Per-bot shares are flat (LUCK 0.18–0.28 across all 8) — no bot is losing to a special mechanism.

**Calibration split of the unattributed bucket** (8 cells, step 24M): the probe self-diagnoses
its selection confound and shows the calibrated-critic signature everywhere (bias_on_wins −2..−13
shaped-V units, bias_on_losses +26..+43). Reweighted to the true ~0.92 win rate the unconditional
bias is ~−5 — **no systematic critic over-valuation vs bots**; the "critic_overvalued ≈ 0.5"
field is the probe's own disclaimed loose upper bound. The unattributed mass reads as genuinely
lost-by-then positions plus lines the shallow 2-alt sweep cannot prove better — consistent with
the floor-leak memory (credit assignment inside the game, not V-level fantasy).

## Measurement 1b — is the model even mispredicting the bots? Yes, exactly as the de-mixing story says

rev-1 TB, tail means ≥20M (the stratified `opp_intent/*_bot` vs `*_pool` metrics):

| metric | vs bots | vs pool |
|---|---|---|
| α top-1 accuracy | 0.471 | 0.505 |
| α info gain over baseline (nats) | **0.048** | 0.278 |
| baseline argmax weight | 0.595 | 0.371 |
| predicted vs actual switch rate | **0.566 vs 0.292** | 0.558 vs 0.401 |

The α head predicts a pool-shaped marginal against everyone: ~0.56 switch mass regardless of
opponent class, which is ~2× the bots' true switch rate, and vs bots it adds almost nothing over
a static baseline (bots being individually MORE predictable — baseline 0.595). `label_bot_frac`
= 0.070: bots are 7% of the intent labels, so the marginal is pool-shaped by construction. At the
probed decisions themselves: α mass on the bot's actual action ≈ 0.25 (uniform would be
~0.15–0.18), α switch mass 0.30 vs realized bot switch rate 0.02–0.08. **The misprediction is
real and has the predicted shape. The next row prices what fixing it is worth.**

## Measurement 2 — the decision-level oracle VoI (the conditioning ceiling)

396 decisions across 225 battles (last 3 eval steps, all 9 bots; strata: the worst-δ crater of
each loss, one uniform-random decision per loss, one per sampled win), 384 scored. Per decision:
the full |our legal| × |opp legal| one-ply grid (mean 5.6–6.8 opp actions) under CRN dice
(`reroll_many`, rust), successors materialized through the real encoder and scored by the
checkpoint's win-prob head. Machinery validated bit-exact: the (chosen, recorded, original-seed)
arm reproduces the recorded next obs `array_equal` and its exact recorded wp.

- **best_marginal** = argmax under the model's own recorded α/β intent read (mapped onto the
  opp's true legal set; belief-unmatched mass spread uniformly)
- **best_oracle** = argmax under the recorded bot action (uniform-over-legal for `random`)
- **VoI** = wp(best_oracle) − wp(best_marginal), both evaluated under the oracle

| stratum | n | flip fraction | VoI (wp) | VoI given flip | oracle gain over CHOSEN | marginal gain over CHOSEN |
|---|---|---|---|---|---|---|
| loss craters | 164 | 0.512 [0.433, 0.592] | **0.082 [0.063, 0.101]** | 0.160 | 0.278 [0.249, 0.307] | 0.196 [0.162, 0.229] |
| loss, random decision | 166 | 0.536 | **0.027 [0.020, 0.035]** | 0.051 | 0.044 | 0.017 |
| win, random decision | 48 | 0.438 | 0.016 [0.007, 0.027] | 0.035 | 0.023 | 0.008 |
| vs `random` bot | 6 | 0.17 | 0.003 | — | 0.010 | 0.007 |

Reading: the best action **changes** at half of all decisions when the opponent model becomes the
oracle — but the flips are cheap. At typical loss decisions the oracle is worth +2.7pp of win
probability; at the crater of a loss, +8.2pp. Translating craters-as-decisive into games: the
oracle-specific term recovers ≈ 8% of losses ≈ **0.6pp of the 7.4pp gap (≈ 92.2%-equivalent)**.
Even granting the FULL one-ply oracle improvement over the played action (0.278) recovers ≈ 28%
of losses ≈ 2.1pp (≈ 93.7%-equivalent) — and 70% of that needs no oracle at all, just one-ply
search under the model's existing marginal.

Per-bot craters: VoI is largest vs `staller_v2` (0.189, n=11) and the two SimpleHeuristics
variants (~0.11) — the bots whose scripted line is most exploitable-if-known — and smallest vs
the aggressive/setup families (~0.05–0.08).

## Measurement 3 — the live oracle-wrapper arm: SIZED, NOT RUN

To detect half-gap closure (+3.7pp from 92.6%, 80% power, α=.05 two-sided): **~550 games/arm
unpaired, ~275 CRN-paired.** Cost: the wrapper needs a live bot-action query (the bot decides
from its own hidden view — in-process both players are available, so this is privileged but
implementable), an in-flight ReconstructionRecord (the `main/search_dividend/record.py`
machinery), and per-decision reroll scoring: ~0.5 day build + ~1–1.5 min/game ⇒ **9–27 h on the
≤2 cores available beside the live run** — over this probe's budget, and the decision-level proxy
already bounds the answer well below the halfway line. Run it only if the candidate survives on
other grounds and the game-level number becomes decision-relevant.

## Registered predictions — scored

1. **"The oracle closes NO MORE THAN HALF the gap (≤95% equivalent)." TRUE, with room to
   spare.** Oracle-specific: ~8% of the gap (~92.2% equiv). Even oracle + one-ply search: ~28%
   (~93.7% equiv). The floor-leak reading holds: credit failures, not opponent-model failures,
   carry the gap.
2. **"Dice share >2%." TRUE.** 7.0% CI [3.5, 11.1] of losses are pure-luck battles
   (unwinnable-at-any-opponent-model under the falsifier's re-roll test); 23% of crater-decision
   mass is bad-tail luck.

## What this licenses (the candidate's second gate)

**The bot-gap motivation for opponent-skill conditioning is DEAD; the candidate is not.** The
de-mixing disease is confirmed exactly as diagnosed (pool-shaped α everywhere, ~zero info gain vs
bots) — but on the bot axis, curing it perfectly at one ply is worth ~0.6pp. Do not build the
skill scalar to fix bot losses. What SI-2 cannot see: the ladder axis (rating-stratified human
deviators), where the same disease may be worth more — that is SI-1's inferability curve plus a
ladder-side VoI, not this probe. Also banked: the risk-note check — the oracle arm's gains at
craters come from picking a different ACTION under the known reply, not from variance-shaving
while ahead (wp_chosen at craters averages 0.31, i.e. behind, where variance-minimization is not
the mechanism).

## Caveats (read before quoting)

- **One-ply, CRN single dice line, scored by the model's own win-prob head.** The Starmie/Ttar
  probe measured that head compressing true structure ~40× in one specimen; a compressed scorer
  compresses the VoI too, so the wp deltas are a floor in scale but the RANKING (oracle vs
  marginal vs chosen, all scored identically) is the load-bearing comparison.
- Craters-as-decisive is an approximation in both directions: a loss can be recoverable at
  non-crater decisions (undercount) and a crater wp-gain need not convert to a win (overcount).
- The α-marginal arm maps believed-move seats onto the true legal set (unmatched belief mass
  spread uniformly) — a charitable-to-α construction; a harsher mapping would only raise the VoI
  slightly, biasing against our conclusion's direction, not for it.
- The falsifier's MISTAKE is proof-positive under a shallow sweep (2 alts, one turn) — the true
  mistake share is higher; that also only strengthens the credit verdict.
- 12 errored decisions (6 corrupt 250-turn-cap records, 4 single-legal-opp, 2 branch desyncs) —
  excluded, none systematic.

## Provenance

- Scripts: `bot_gap_oracle_voi_probe.py` (the grid probe), `bot_gap_oracle_voi_aggregate.py`,
  `bot_gap_falsify_sweep.sh` (this directory). Raw cells/rows regenerable in `tmp/si2/` (gitignored).
- Machine record: `bot_gap_oracle_voi_2026-08-31.json`.
- TB tags: `opp_intent/*_bot` / `*_pool` tail means ≥20M from rev-1's own events.
- Cluster bootstrap over battles throughout (the Simpson-trap rule); 2000–3000 resamples.
