# X5 Tier 0: the oracle counterfactual on OTHER's physics (2026-10-03)

**The question.** M3 = (c) gives X5's OTHER token the blob's AVERAGED attacker and defender over the renormalised tail
of unnamed species, at a 1:1 budget (one named hypothesis per unseen seat). This measurement bounds how much better
handling of OTHER could buy on the physics channel before X5 is built. It also asks whether a larger hypothesis budget
is worth its cost. **This is a measurement only:** CPU, no model change, no training, no checkpoint, and nothing
under `models/` is read.

## Verdict

- **Keep (c) at 1:1 for the A/B.** On real decision states, (c) at 1:1 already gets most of what any budget can reach:
  - **76–100 %** of the gap between the blob (R0) and the no-averaging limit (Rinf) on the four threat proxies;
  - **75 %** of that gap on the threat error itself.
- **The budget lever, (c) + 6 tokens (budget (6 − r) + 6), is small.**
  - Paired against (c) at 1:1 on the same decisions, it cuts:
    - the race-proxy flips by **−0.97 pp** [−1.34, −0.65];
    - the expected-threat error by **−2.9 pp of our HP** [−3.2, −2.5].
  - It reaches ≥ 92 % of the gap on every threat read; budgets 12 and 18 add almost nothing beyond it.
  - It slightly WORSENS every defender-side read: out +0.46 pp, ko +0.70 pp, best-move flips +0.34 pp.
  - Cost: ≈ +7 % forward FLOPs (ESTIMATED, design §3.6). It is the named follow-up lever, not an A/B change.
- **The large "perfect OTHER" ceiling is INFORMATION, not physics.**
  - A mass-conserving perfect OTHER (OP: OTHER priced as the true missed mons) removes 43–71 % of (c)'s feature error
    at 1:1. It would cut best-move flips from 23.5 % to 8.8 % and race flips from 9.4 % to 3.3 %.
  - Only the AVERAGING part of that is buyable by budget or physics: the Jensen share is ≤ 33 % on threat reads and
    ≤ 0 on defender reads.
  - The rest needs a sharper BELIEF: which species are in the tail.
- **The M2 = C worst-case channel does not respond to budget at all.**
  - `in_max` stays at 22.9–23.2 pp, and P2 flips stay at 8.1–8.2 %, from 1:1 to the no-averaging limit.
  - Under M2 option A (noisy-OR expected max), the same read responds: 16.1 pp at 1:1 → 12.4 pp at k + 6.
  - So for worst-case threats, M2 binds before M3's budget does (FINDING F3).

## Data and method

- **Decisions.** The M5 Lane S bank (`../m5_laneS/bank_v1`, content sha `8ca1bfa5…`): 20,712 decisions in 580 battles.
  - **G-4 is VERIFIED for this bank.** Every battle record carries BOTH packed teams.
  - The observations are RE-ENCODED through the Rust core (`main.policy_spectrum.reader.reencode`). All **19,964**
    rows that carry a recorded sha re-encode byte-equal (`rows_gate` in `out/tier0.json`).
  - **Used: 13,853 decisions in 580 battles.** The remaining 6,859 were dropped:
    - 5,374 had no hidden opponent mon;
    - 1,485 were gated: no opponent active, or our active fainted (the op zeroes those rows).
  - By reveals r = 1 / 2 / 3 / 4 / 5: 2,352 / 2,075 / 2,574 / 3,262 / 3,590 decisions.
  - By opponent class: bot 7,622 · pool snapshot 5,738 · exploiter 493.
- **On-pool only.** All 580 opponent teams are pool archetypes (`data/teams/gen3_team_archetypes.json`, by species
  set). **Off-pool is TODO**: no ladder-team decision states with both teams are banked.
- **The op is the production op.** It runs on a FRESH production-surface learner (`learner_golden.build_learner`,
  seeded, one thread), so every learned belief delta is zero-init and every belief is the cold-start Smogon one. The
  op's hidden-species input (`T0SpeciesPrior`) is parameter-free, so **R0 is exactly what production computes,
  whatever the checkpoint.**
- **Kernels.** `scripts/kernels.py` re-spells two production kernels, vectorised over every species at once.
  `scripts/parity.py` checks them against the real op functions on 512 banked states, and all seven checks are
  **bit-exact (max |Δ| = 0.0)** (`out/parity.json`):
  - `_outgoing_matrix` hidden columns: R0 itself, one-hot overrides, and a tail override;
  - `pairwise_bench_incoming` (D4): fake-revealed hypothesis attackers, and REAL revealed bench attackers under the
    production cold-start move posterior;
  - SPECIES_TYPE against the observed types.
- **The one construction with no production counterpart** is OTHER's averaged ATTACKER (`kernels.averaged_attacker`;
  FINDING F4). Today every hidden attacker is gated off. It is defined as:
  - E[atk] and E[spa] under the tail;
  - today's hidden-slot composed move posterior (the E10 mixture) as its moves;
  - presence-weighted expected STAB.
- **Beliefs** (`scripts/proxies.py`). These are the revision's three, re-built from `m1m3.py`; the fitted
  constants are asserted equal to `m1m3.log` at run time:
  - the Smogon prior (cold start);
  - the in-sample pool naive-Bayes proxy;
  - the exact in-sample memory proxy.

  π is the logistic fixed-size construction (§3.2), in fp64. Hypotheses are the top-H species by π, taken in one
  stable order.
- **Representations.**

| tag | what the hidden opponent mons are priced as |
|---|---|
| R0 | today's blob: every hidden seat = the averaged defender over the belief marginal; hidden attacker gated off (threat 0); P(KO) nulled |
| R1_H | X5 + M3 (c): top-H named mons (weight π, exact physics) + OTHER (mass k − Σ named π) as the averaged defender AND attacker over the renormalised tail; OTHER's P(KO) nulled |
| R2_H | X5 + M3 (d): the same named list; OTHER has no physics |
| OP_H | **perfect OTHER physics, mass kept**: OTHER priced as the TRUE hidden mons the list missed (their mean; their worst member for max reads). `1 − err(OP)/err(R1)` = the share of R1's error OTHER's tail representation causes |
| OO_H | the literal oracle: missed true mons at weight 1 each. Not mass-conserving; the named list's excess mass equals OTHER's deficit, so the two errors partly cancel in R1 and OO can read worse than R1 at H > k. A bound, not a decomposition |
| Rinf | every species named, no averaging anywhere: the floor ANY budget can reach under the belief |
| T | the truth: the actual hidden species (π = 1), through the same op path |

  Budgets: 1:1 (= k = 6 − r), k + 6, 12, 18.
- **Features**, per entity, then summarised over the hidden set:
  - **out:** our move's max-roll damage on a full-HP switch-in, % of its max HP.
  - **ko:** acc × P(OHKO) on that switch-in.
  - **in:** its worst hit on our active, % of our max HP (D4).
  - **bulkP / bulkS:** E[maxhp] × E[def] or E[spd], as a relative error %.
  - **margin:** best out × our HP fraction − in.

  Each feature is summarised as the presence-weighted MEAN (`in_mean`, and the others). The `in` feature also gets two
  max-type reads: `in_max`, the presence-scaled MAX (M2 = C, with OTHER's presence capped at 1, FINDING F4), and
  `in_emax`, the noisy-OR expected max (M2 option A, for comparison).
- **Decision proxies.** A proxy FLIPS when the representation's answer differs from the truth's.
  - **P1:** the best move by mean out, among ≥ 2 usable damaging moves.
  - **P2:** safe to stay in, `in_max` < our HP.
  - **P2e:** the same with `in_mean`.
  - **P2a:** the same with `in_emax`.
  - **P3:** the KO race, margin > 0.
- **Rule 8.** A decision is EXCLUDED from a proxy when the truth's OR the representation's value lies within
  **1 pp of HP** of the boundary (P1: a top-1 − top-2 gap < 1 pp). Per-cell excluded shares are in the JSON.
  - P1 is also structurally excluded on **31.1 %** of decisions (fewer than 2 usable damaging moves).
  - The flip comparisons below use the **COMMON included set**: decisions that no representation excluded, so every
    representation is read on the same decisions.
- **Intervals.** Battle-clustered bootstrap: 400 resamples, seed 0, 580 clusters.
- **Reproducible.** Every script is seeded, and two full runs give byte-equal `tier0.json`, `rows.csv.gz` and
  `decisions.csv.gz`.

```bash
cd designs/research_state/measurements/x5_tier0_oracle_2026-10-03/scripts
export PYTHONPATH=<checkout>/src:.
python parity.py          # ~1 min; writes out/parity.json, exits 1 on any mismatch
python tier0.py           # ~2 min on 4 threads (+ ~1 min the first time to re-encode the bank; cached in /tmp)
python table.py           # prints the tables below from out/tier0.json
```

## Results (Smogon prior; 13,853 decisions; battle-clustered 95 % CI)

**(a) Feature error against the truth.** Mean |error|, in pp of HP (bulk: relative %).

| rep | out | ko | in_mean | in_max (M2 C) | in_emax (M2 A) | bulkP | bulkS | margin |
|---|---|---|---|---|---|---|---|---|
| R0 (blob) | 15.2 [14.4, 16.2] | 7.6 | 24.1 [22.9, 25.1] | 35.2 | 35.2 | 15.3 | 19.4 | 21.5 |
| R1 1:1 (c) | 15.4 [14.7, 16.4] | 7.9 | 12.0 [11.5, 12.7] | 22.9 | 16.1 | 15.6 | 19.4 | 14.5 |
| R1 k+6 | 15.9 | 8.6 | 9.2 [8.7, 9.7] | 23.1 | 12.4 | 15.7 | 19.8 | 13.6 |
| R1 12 | 16.0 | 8.9 | 8.9 | 23.2 | 11.9 | 15.7 | 19.9 | 13.4 |
| R1 18 | 16.1 | 9.1 | 8.4 | 23.2 | 11.3 | 15.7 | 20.0 | 13.1 |
| Rinf | 16.3 | 9.5 | 8.1 [7.6, 8.7] | 23.2 | 10.9 | 15.7 | 20.0 | 12.9 |
| R2 1:1 (d) | **28.1** | 7.9 | 16.5 | 23.7 | 18.4 | **65.6** | **66.9** | 21.3 |
| OP 1:1 (perfect OTHER) | 8.8 | 3.9 | 4.7 | 6.7 | 5.9 | 10.4 | 11.6 | 7.0 |
| OO 1:1 (literal oracle) | 11.4 | 3.0 | 7.1 | 5.5 | 5.2 | 27.4 | 26.3 | 8.2 |

**(b) The share of (c)'s error that comes from OTHER.**
- **OTHER share:** 1 − err(OP)/err(R1).
- **Jensen share:** 1 − err(Rinf)/err(R1), the part a budget or exact physics on the same belief can remove. The
  rest of the OTHER share is information.

| feature | OTHER share, 1:1 | k+6 | 12 | 18 | Jensen share, 1:1 |
|---|---|---|---|---|---|
| out | 43 % [40, 46] | 7 % | 5 % | 2 % | **−5 %** |
| ko | 50 % | 14 % | 9 % | 3 % | **−20 %** |
| in_mean | 61 % [58, 63] | 17 % | 12 % | 2 % | 33 % [28, 37] |
| in_max (M2 C) | 71 % | 31 % | 24 % | 9 % | −1 % |
| in_emax (M2 A) | 63 % | 18 % | 13 % | 5 % | 32 % |
| bulkP / bulkS | 33 % / 41 % | −9 % / 11 % | −8 % / 11 % | −9 % / 7 % | −1 % / −3 % |
| margin | 52 % [49, 55] | 12 % | 7 % | 2 % | 11 % |

**(c) Decision flips on the common included set** (%; the n is per proxy):

| rep | P1 best move (n 6,153) | P2 safe, M2 C (13,511) | P2e safe, mean (12,920) | P2a safe, M2 A (13,060) | P3 KO race (9,856) |
|---|---|---|---|---|---|
| R0 (blob) | 23.2 | 10.5 | 4.0 | 9.0 | 18.8 |
| R1 1:1 (c) | 23.5 | 8.1 | 2.0 | 4.3 | 9.4 |
| R1 k+6 | 23.8 | 8.2 | 1.4 | 3.3 | 8.4 |
| R1 12 | 23.8 | 8.2 | 1.4 | 3.3 | 8.3 |
| R1 18 | 23.8 | 8.2 | 1.4 | 3.3 | 8.3 |
| Rinf | 23.8 | 8.2 | 1.3 | 3.3 | 8.3 |
| R2 1:1 (d) | 29.9 | 8.3 | 2.9 | 5.0 | 10.0 |
| OP 1:1 (perfect OTHER) | 8.8 | 2.6 | 1.0 | 1.7 | 3.3 |

Paired deltas against R1 1:1, in pp [95 % CI]:

| proxy | k + 6 | Rinf | OP 1:1 |
|---|---|---|---|
| P1 | +0.34 [+0.08, +0.66] | +0.34 [+0.08, +0.66] | −14.7 |
| P2 | +0.05 [−0.05, +0.14] | +0.07 [−0.02, +0.16] | −5.5 |
| P2e | −0.58 [−0.78, −0.41] | −0.66 [−0.87, −0.46] | −1.0 |
| P2a | −1.00 [−1.23, −0.76] | −1.06 [−1.32, −0.82] | −2.6 |
| P3 | −0.97 [−1.34, −0.65] | −1.08 [−1.46, −0.73] | −6.1 |

**(d) The recovery curve.** The share of the R0 → Rinf gap each budget recovers. Rinf is the best any budget can do
under this belief.

| read | 1:1 | k + 6 | 12 | 18 | R0 → Rinf (of R0's own error) |
|---|---|---|---|---|---|
| in_mean error | 75 % | 93 % | 95 % | 98 % | 66 % |
| in_emax error | 79 % | 94 % | 96 % | 98 % | 69 % |
| margin error | 81 % | 92 % | 95 % | 98 % | 40 % |
| P2e flips | 76 % | 97 % | 97 % | 97 % | 67 % |
| P2a flips | 82 % | 99 % | 99 % | 100 % | 64 % |
| P3 flips | 90 % | 99 % | 100 % | 100 % | 56 % |
| P2 flips (M2 C) | 103 % | 101 % | 101 % | 100 % | 22 % |
| out / ko / bulk / P1 | — (R0 is already at or below Rinf; no budget helps) | | | | ≤ 0 % |

The cheapest budget that recovers ≥ 80 % of the gap on EVERY threat read is **k + 6**. At 1:1, (c) clears 80 % on 4 of
the 7 reads (P2, P2a, P3, margin) and reaches 75–79 % on the other three.

**By reveals** (prior; in_mean pp / P2e % / P3 %; included-set flips):
- **r = 1–2:** a budget keeps paying beyond k + 6. At r = 1, in_mean reads 11.3 at 1:1, 8.0 at k + 6, 6.0 at 18 and
  4.7 at Rinf.
- **r ≥ 3:** k + 6 is already at Rinf (r = 4: 8.8 = 8.8).
- **Late game (r = 5) has the largest residual error in every representation** (in_mean 15.5 at 1:1, 13.0 at Rinf).
  The last unseen mon is the hardest to guess.

**Memorising proxies.** These are fitted to the banked head's NLL. Every pooled number is within ≈ 0.5 pp of the
prior's. For example, `memo_exact`:
- R1 1:1: in_mean 11.6, P3 14.0 %;
- k + 6: 8.6, 12.1 %;
- OP 1:1: 4.8, 3.6 % (included-set rates).

They are not sharper than the prior on these states (F6), so Tier 0 says nothing about a much sharper trained belief.

**OTHER mass on real states** (prior). This is the share of hidden mass and the recall of the true hidden mons by the
named list:

| budget | r = 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1:1 | 66 % · 0.30 | 65 % · 0.33 | 65 % · 0.27 | 67 % · 0.26 | 74 % · 0.14 |
| k + 6 | 39 % · 0.50 | 35 % · 0.60 | 32 % · 0.60 | 28 % · 0.59 | 25 % · 0.69 |

The mass matches the revision's synthetic rows, but recall is LOWER on real states (F2).

## Recommendation for M3

1. **Keep (c) at 1:1 for the A/B (recommended).**
   - It is never worse than the blob on the defender channel: out 15.4 vs 15.2, bulk equal, P1 23.5 vs 23.2.
   - It halves the blob's threat error: 24.1 → 12.0.
   - It cuts the KO-race flips from 18.8 % to 9.4 %.
   - It recovers 76–100 % of what any budget could on the threat proxies.
   - (d) is confirmed a regression on the defender channel: out 28.1, bulk 66 %, P1 29.9 %.
2. **(c) + budget k + 6 is the follow-up lever, and the data price it.**
   - Gain: ≈ −1 pp of race and E-max safety flips, −0.6 pp of mean-safety flips, and −2.9 pp of HP on the
     expected-threat read.
   - Loss: ≈ +0.3–0.8 pp on the defender reads.
   - Cost: +6 tokens ≈ +7 % forward FLOPs (ESTIMATED), op rows ∝ budget (NOT MEASURED), +0.5 agent-day.
   - Budgets 12 and 18 buy ≤ 0.1 pp more on every flip proxy. Only at r = 1–2 does 18 still help the threat read.
3. **What the data point to instead, both larger than the budget.**
   - **(i) Belief sharpness.** It is the bulk of the OP ceiling: up to −14.7 pp of best-move flips and −6 pp of race
     flips. It would be served by δ_θ learning (already in X5), not by OTHER's physics.
   - **(ii) M2 option A (noisy-OR).** On the worst-case channel it is worth −1.9 pp of safety flips at 1:1 (6.3 %
     vs 8.1 % on the 13,532 decisions both proxies include) and cuts the worst-hit error from 22.9 to 16.1 pp.
   - Both are already the named post-X26 levers in `design_x5_tradeoffs.md`. This read says to promote M2 → A ahead of
     the budget lever if only one is tested.

**What Tier 0 CANNOT tell us.**
- It measures the INFORMATION in the op's physics channel under simple read-outs, not strength.
- The network reads per-seat tokens through attention and may extract more from named hypotheses, or less, than a mean
  or max summary.
- The decision proxies are hand rules, not the policy.
- The closest strength evidence cuts against expecting much: a privileged critic that SAW the true opponent team
  bought "no resolution and no opponent conditioning" (L17103).
- Even the perfect-OTHER ceiling here is an upper bound on information, not a prediction of win rate.

## Limitations

- **Cold-start beliefs only.** The memorising proxies are species-only and, fitted to the head's NLL, no sharper than
  the prior here. A trained δ_θ could be sharper: OTHER's share would shrink, and so would the Rinf → OP gap.
  UNVERIFIED.
- **On-pool only.** The bank is loss-enriched by design and is 55 % bot games.
- **Off-pool and the opponent ACTIVE's move axis (OTHER_move) are TODO.**
- **The truth is the true SPECIES through the op's own approximations**, with sets not modelled. Those approximations:
  - de-timid offense, spread-prior bulk, and an expected ability;
  - no opponent items in this path;
  - full-HP switch-ins;
  - P(KO) nulled on the hidden path.

  So "truth" is the X5-reachable species truth. Set-level uncertainty (EVs, moves, items) is out of scope.
- **Hypothesis seats are priced on the hidden path** (SPECIES_SPREAD_PRIOR bulk, expected ability). X5 will price
  them on the revealed path with the spread head's row for that species. At cold start the two are the same species
  prior row in different spellings, NOT checked equal here.
- **The P3 race ignores speed and integer hit counts. `ko` partly measures the nulling rule**, since R0 and OTHER are
  nulled as built while hypotheses and the truth are not.
- **R0's hidden-threat channel is 0 in the op.** The E5 tail seats do carry a crude, type-blind BP read of hidden mons'
  move posteriors; it is not priced against our active and is not counted here.

## FINDINGS

- **F1. G-4 VERIFIED for the Lane S bank.** Both packed teams sit on every battle record, and all 19,964 rows with a
  recorded sha re-encode byte-equal. All 580 opponent teams are pool archetypes.
- **F2. On real states the 1:1 list recalls only 14–33 % of the true hidden mons under the prior,** against 34–43 % on
  the revision's random-reveal rows, at the same OTHER share (65–74 %).
  - Real reveal order is not random: what stays hidden is what the prior guesses worst.
  - The revision's recall numbers (§3.3) are optimistic for play.
- **F3. Under M2 = C, the worst-case channel is budget-blind:**
  - `in_max` 22.9 → 23.2 pp and P2 8.1 → 8.2 % from 1:1 to Rinf;
  - presence scaling floors it.

  M2 option A responds to budget (16.1 → 10.9 pp) and cuts the error at 1:1 by 30 % (22.9 → 16.1 pp). This is a direct read for
  `design_x5_tradeoffs.md` §1 "how we will know it is binding".
- **F4. Two unspecified pieces of M3 (c) / M2 that U3 must decide:**
  - how OTHER's AVERAGED ATTACKER is built: moves, STAB, offense. This measurement declared one in
    `kernels.averaged_attacker`;
  - how OTHER's mass, which exceeds 1 on 73.6 % of decisions at 1:1 (37.5 % at k + 6), enters a class-M (max)
    site. This measurement capped its presence at 1.
- **F5. (d) is confirmed a regression on the defender channel.** At 1:1 its out error is 28.1 against the blob's 15.2,
  its bulk error 66 % against 15 %, and its best-move flips 29.9 % against 23.2 %.
- **F6. The banked-head-fitted memorising proxies are not sharper than the Smogon prior on these states.** The live
  head's π and its OTHER mass are still UNVERIFIED.
- **F7. Larger budgets slightly WORSEN the defender-side mean reads.** Against 1:1, k + 6 adds out +0.46 pp
  [+0.34, +0.60], ko +0.70 and bulkS +0.39.
  - The averaged defender's shrinkage beats the exact expectation under an uncertain belief (the Jensen share is
    negative).
  - The "Blissey + Skarmory blend" harm the note expects is not visible in these reads.
- **F8. GIGO, out of this unit's scope (reported, not fixed): OUR fixed-damage moves are priced as UNUSABLE in every
  outgoing op block.**
  - `usable = legal * (bp > 0)` (`damage_op_blocks.py:163, 320, 462`), and Seismic Toss, Night Shade, Dragon Rage and
    Sonic Boom have `MOVE_BP` 0. So `fixed = MOVE_FIXED_DAMAGE * usable` is always 0, and the fixed-damage override
    in those blocks is dead code.
  - On this bank: **1,975** (decision, legal fixed-damage move of ours) pairs, **0** priced above zero in
    `_outgoing_matrix`.
  - The incoming side is correct; the existing test, `damage_op_test.py:1200`, covers only that side.
- **F9. `tier0.py` is registered in `src/measurements_readout_gate_test.py`** (the P0 unit's precedent, floor 6 → 7).
  Its `--check` resolves the bank, the pool archetypes, the m1m3 log and the sibling scripts.

## Files

| path | what |
|---|---|
| `scripts/common.py` | bank load + re-encode (cached in `/tmp/x5tier0_cache`), true teams, the fresh extractor |
| `scripts/proxies.py` | the three beliefs + the fixed-size construction (from `x5_revision_2026-10-03/scripts/m1m3.py`) |
| `scripts/kernels.py` | the outgoing and D4 kernels, vectorised over species; the averaged defender and attacker |
| `scripts/parity.py` | the seven parity checks against the production op → `out/parity.json` |
| `scripts/tier0.py` | the measurement → `out/tier0.json`, `out/rows.csv.gz`, `out/decisions.csv.gz`; `--check` for the readout gate |
| `scripts/table.py` | prints the tables above from `out/tier0.json` |
| `out/tier0.json` | every cell: per belief × representation × r, mean / CI / median / p90 per feature, flip rate + CI + excluded share per proxy; derived OTHER / Jensen shares and recovery with CIs; common-set flips with paired deltas; belief mass and recall |
| `out/rows.csv.gz` | per decision × representation, Smogon-prior belief: the eight feature errors and the five proxy flips (−1 = excluded). The memorising proxies' rows are not committed (aggregated in `tier0.json`; `tier0.py` regenerates them) |
| `out/decisions.csv.gz` | the decision index: bank id, battle index, r, opponent class, our HP fraction, usable damaging moves |
