# THE MIXTURE DIAGNOSTIC — does arm A's win-prob critic know WHO it is playing?

`ai_v12_02_winprob_critic`, last four trace cycles (50.0M · 60.0M · 70.0M · 74.0M).
Run 2026-09-08, offline, CPU only, nothing written under `models/`. Reproduce with `./run.sh`.

**VERDICT: DETECTED.** The critic emits close to one marginal win probability per state
regardless of opponent. On turn-1–3 states — where the board has said nothing and the opponent
is the only thing that separates outcomes — the between-opponent spread of `V` is
**0.334× [0.318, 0.451]** the between-opponent spread of the outcome it is supposed to predict
(delta **−0.0731 [−0.0840, −0.0588]**, battle-clustered, CI clear of zero). The head's turn-1
opinion ranges over 0.685–0.744 across fourteen opponents whose true win rates range over
0.635–1.000.

---

## 1. The question, and the prediction stated before the reading

`V` is trained against a MIXTURE: nine scripted bots, five pool sentinels. If the head never
learned to condition on the opponent it emits the mixture's marginal — too LOW against a weak
opponent (whose true rate is ~1.0) and too HIGH against a strong one. Two predictions follow, and
the second needs no strength axis at all:

| | mixture head | separating head |
|---|---|---|
| **A · slope of bias on opponent strength.** bias := `V` − outcome. Weak opponent: outcome ≈ 1.0 > V ⇒ bias NEGATIVE. Strong: outcome ≈ 0.5 < V ⇒ bias POSITIVE. | **POSITIVE slope** | slope ≈ 0 |
| **B · between-opponent spread.** For ANY calibrated critic `E[V \| opponent] = E[y \| opponent]` exactly, so the two spreads must be EQUAL. | **sd(V) ≪ sd(outcome)** | ratio ≈ 1 |

Test B is the sharper one: it is an identity, not a regression, and it cannot be argued with by
re-scaling the strength axis.

## 2. The frame

| | |
|---|---|
| states / battles / cells | **29,495** / **951** / 56 (14 opponents × 4 cycles) |
| selection | every cycle carries an `eval_manifest.json` with `selection_schema: 1` — **no cycle is SELECTION UNKNOWN**. The quota is 10 losses + 5 wins per opponent, so the traced win rate is ~0.50 against a true 0.83; every statistic here is Horvitz-Thompson reweighted by `capture_rate_win` / `capture_rate_loss` |
| QC — is `V` the win-prob head? | `max\|values − win_probs\| = 0.0` exactly across all 29,495 states |
| QC — does the reweighting recover the population? | the IPW outcome equals the manifest's `battles_won / battles_played` **exactly** in all 56 cells (the HT identity), so the reweighted outcome IS the cycle's true 100-game win rate |
| strength axis, bots | `data/gen3_bot_elo_anchors.json` — 2,000–2,700 games per pair, independent of this run |
| strength axis, sentinels | `eval_results.jsonl`'s per-cycle `sentinels[k].step` → an **all-steps refit** of the snapshot ladder. The positional map `sentinel_k` → k-th eval-row entry is **VERIFIED, not assumed**: the eval row's win rate equals the manifest's own `battles_won/100` to the digit in all 20 sentinel cells |

**Why a refit and not the committed `ladder.json`.** The committed file rates only 36M–74M — pool
grooming deleted the early snapshots and the fit was sliced to the survivors (the P1 defect named
in UNDERSTANDING §4.2b) — which would leave 7 of the 20 sentinel cells unrated. `games.jsonl`
still carries dense frozen-vs-frozen edges from 4.0M up, so `refit_ladder.py` rates all 36 nodes
on one bot-anchored scale, and this tree's `fit_ladder` drops the 165 greedy-vs-stochastic
sentinel edges that made the committed file read high. Final node: **2015.7** refit vs **2057.3**
committed.

## 3. Per-opponent, averaged over the four cycles

| opponent | Elo | battles | states | true WR | traced WR (raw) | reweighted | mean V | **bias** | 95 % CI | V on turn 1–3 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 1000.0 | 32 | 835 | 1.000 | 1.000 | 1.000 | 0.908 | **−0.092** | [−0.117, −0.070] | 0.739 |
| aggressive | 1511.7 | 59 | 1318 | 0.918 | 0.547 | 0.917 | 0.874 | −0.044 | [−0.110, +0.021] | 0.745 |
| staller_v2 | 1554.9 | 58 | 2360 | 0.925 | 0.557 | 0.925 | 0.858 | −0.067 | [−0.143, +0.003] | 0.733 |
| staller | 1570.9 | 56 | 1719 | 0.922 | 0.573 | 0.922 | 0.886 | −0.037 | [−0.101, +0.026] | 0.741 |
| heuristic | 1577.5 | 68 | 2074 | 0.880 | 0.473 | 0.880 | 0.867 | −0.013 | [−0.094, +0.062] | 0.738 |
| setup_sweep | 1597.8 | 68 | 1713 | 0.870 | 0.481 | 0.870 | 0.848 | −0.022 | [−0.105, +0.057] | 0.712 |
| setup_sweep_v2 | 1618.7 | 70 | 1636 | 0.890 | 0.458 | 0.890 | 0.859 | −0.031 | [−0.112, +0.043] | 0.741 |
| aggressive_v2 | 1630.1 | 71 | 1834 | 0.882 | 0.452 | 0.882 | 0.844 | −0.039 | [−0.126, +0.040] | 0.712 |
| heuristic2 | 1638.8 | 69 | 1773 | 0.890 | 0.469 | 0.890 | 0.857 | −0.033 | [−0.116, +0.042] | 0.729 |
| sentinel_4 | 1907.0 | 80 | 2904 | 0.745 | 0.400 | 0.745 | 0.744 | −0.001 | [−0.108, +0.100] | 0.693 |
| sentinel_3 | 1951.1 | 80 | 2843 | 0.718 | 0.400 | 0.718 | 0.760 | +0.043 | [−0.063, +0.143] | 0.689 |
| sentinel_2 | 1983.2 | 80 | 2959 | 0.715 | 0.400 | 0.715 | 0.731 | +0.016 | [−0.098, +0.122] | 0.688 |
| sentinel_1 | 2004.9 | 80 | 2799 | 0.635 | 0.400 | 0.635 | 0.741 | **+0.106** | [−0.015, +0.218] | 0.683 |
| sentinel_0 | 2014.5 | 80 | 2728 | 0.653 | 0.400 | 0.653 | 0.753 | **+0.100** | [−0.019, +0.209] | 0.694 |

The outcome column moves through **0.365** of win rate. The turn-1–3 `V` column moves through
**0.062**. Per-cell intervals are in `mixture_stats.json` (`cells`); the pooled tests below are
what carry the claim, because a per-cell CI at ~17 battles cannot resolve a ±0.05 effect.

## 4. Test A — the slope of bias on opponent strength (per 100 Elo, cycle fixed effects)

| variant | slope | 95 % CI | reading |
|---|---|---|---|
| **all 14 opponents**, battles clustered | **+0.0171** | **[+0.0131, +0.0209]** | **DETECTED** |
| **all 14**, battles **and opponents** resampled | +0.0178 | **[+0.0094, +0.0300]** | **DETECTED** |
| bot-only, battles clustered | **+0.0098** | **[+0.0059, +0.0127]** | **DETECTED** (bots as the fixed pinned set they are) |
| bot-only, battles **and opponents** resampled | +0.0106 | [−0.0177, +0.0447] | **NOT DETECTED** |

🚨 **The two bot-only rows are not in conflict; they answer different questions.** The nine bots
are a FIXED pinned roster, not a sample, and against that roster the slope is detected. Treating
them as one draw from a population of possible bots is the conservative reading, and it is
underpowered by construction: eight of the nine sit inside 127 Elo of each other (1511.7–1638.8),
so almost all of the bot-only leverage is `random` versus the rest. **The 14-opponent row is the
one to quote**, and it survives both resamplings.

Over the full strength range the slope is the whole effect: 1031 Elo × 0.0171 = **+0.176** of
bias, against an observed −0.092 → +0.106 swing.

## 5. Test B — between-opponent spread of V vs of the outcome (the identity)

Each spread is corrected for its own sampling noise before comparison: the outcome's per-opponent
mean carries exactly `p(1−p)/100`, V's carries its own squared standard error. Without that
correction a between-group variance is inflated by group size and the comparison would be about
cell counts, not about opponents. Cells are weighted equally.

| states used | sd(V) | sd(outcome) | **ratio** | 95 % CI | delta | 95 % CI | |
|---|---|---|---|---|---|---|---|
| **turn 1 only** | 0.0339 | 0.1098 | **0.308** | [0.297, 0.431] | −0.0759 | [−0.0865, −0.0610] | **DETECTED** |
| **turn 1–3** | 0.0366 | 0.1098 | **0.334** | [0.318, 0.451] | **−0.0731** | **[−0.0840, −0.0588]** | **DETECTED** |
| early (≤10) | 0.0527 | 0.1098 | 0.481 | [0.435, 0.576] | −0.0570 | [−0.0700, −0.0452] | DETECTED |
| mid (11–24) | 0.0752 | 0.1098 | 0.685 | [0.605, 0.826] | −0.0346 | [−0.0485, −0.0187] | DETECTED |
| late (≥25) | 0.1570 | 0.1112 | 1.412 | [0.995, 1.754] | +0.0458 | [−0.0006, +0.0856] | descriptive — see below |
| **all states** | 0.0629 | 0.1098 | **0.573** | [0.509, 0.689] | −0.0469 | [−0.0604, −0.0334] | **DETECTED** |
| all states, **bot-only** | 0.0183 | 0.0327 | 0.560 | [0.450, 0.992] | −0.0144 | [−0.0269, −0.0003] | DETECTED, marginally |
| turn 1–3, **bot-only** | 0.0218 | 0.0327 | 0.667 | [0.535, 1.139] | −0.0109 | [−0.0222, +0.0052] | NOT DETECTED (underpowered: the bots span only 0.033 of win rate) |

⚠️ **The late row is DESCRIPTIVE, and its ratio > 1 is not a rebuttal.** By turn 25 the board
itself says who is winning, so `V` legitimately spreads; and the comparison is no longer
like-for-like, because the outcome column is the UNCONDITIONAL per-opponent win rate while the
`V` column is conditioned on having reached turn 25. The rows that carry the claim are turn-1 and
turn-1–3, where the bucket contains every battle (each battle has a turn 1) and the two sides
condition on the same event.

**The head is not merely flat.** On turn-1–3 states its per-battle spread is large — within
opponent **sd 0.0948**, and 0.0703 across the 51 trainee teams with ≥4 battles — against a
between-opponent **sd 0.0229** (within/between = **4.1×**). It has the dynamic range at turn 1;
it spends it on its own team and lead, not on who is across the table.

**η² over the traced state population**, for completeness (`var_decomp` in the JSON): the outcome
puts 0.102 of its variance between opponents, `V` puts 0.147 of a variance that is 5× smaller —
`btw_y` 0.0169 / `wth_y` 0.1480 against `btw_V` 0.0052 / `wth_V` 0.0300. ⚠️ These four are
**uncorrected** and their bootstrap intervals sit above their point estimates (the standard
upward bias of a resampled between-group variance) — read them as descriptive; §5's corrected
table is what the verdict rests on.

## 6. Counter-hypotheses

**(a) The loss-preferring capture rate.** Handled by reweighting, and it matters enormously: the
traced sample's own win rate is **0.501** against a true **0.832**. Computed end to end with no
reweighting at all the bot slope is **+0.0607 [+0.0540, +0.0675]** — six times the reweighted
+0.0098. **Reweighting SHRINKS the effect; the naive read overstates it.** Every headline above is
the conservative, reweighted one. (SURVIVES as a hazard for anyone reading the raw tree; does not
threaten this reading.)

**(b) The greedy-vs-stochastic sentinel regime.** Arm A ran the OLD regime, worth +8.9 pp
[+7.0, +10.7] to the trainee, so the sentinels' recorded win rates are INFLATED — and since
bias = `V` − outcome, an inflated outcome at high strength SHRINKS a positive slope. Correcting
the five sentinel cells by −8.9 pp moves the 14-opponent slope from +0.0171 to
**+0.0300 [+0.0261, +0.0338]**. **The handicap works against the finding.** The bot stratum is
untouched by it, and the bot-only slope is reported separately above. **ELIMINATED as an
explanation.**

**(c) Late-game states dominating the count.** They do not carry it. The bot-only slope is
**+0.0169 [+0.0120, +0.0214]** on early states, +0.0076 [+0.0036, +0.0103] on mid, and
**−0.0095 [−0.0249, −0.0031]** on late — the effect is strongest exactly where the board tells
you least, and reverses where it tells you most. Test B's tercile rows say the same thing in the
other direction (ratio 0.33 early → 1.41 late). **ELIMINATED.**

**(d) A per-TEAM bias masquerading as a per-opponent bias.** Partly SURVIVES, and it is a finding
of its own. Measured as between-group variance of the residual `V − y` in EXCESS of a
label-permutation null (which prices the small-group inflation: 216 teams over 951 battles is
~4.4 battles each, against 14 opponents at ~68): opponent excess **+0.0027 [+0.0013, +0.0074]**,
team excess **+0.0246 [+0.0202, +0.0385]**, gap **−0.0220 [−0.0345, −0.0164]**. **The trainee's
own team explains ~9× more of the critic's residual than the opponent does.** But it does not
explain the slope: trainee teams are drawn independently of which opponent is queued, so a team
effect adds noise to a cell mean rather than tilting it against strength, and the battle-clustered
bootstrap already prices that noise. The honest statement is that this read finds **two**
conditioning failures, one of which it was not looking for.

**(e) Was any cycle SELECTION UNKNOWN?** No. All four carry a schema-1 manifest with all 14
opponent rows.

## 7. Per cycle — the defect is stationary, not a run-end artefact

| cycle | slope (14 opp) | slope (bot) | sd(V)/sd(outcome), all states | turn 1–3 |
|---|---|---|---|---|
| 50,000,016 | +0.0179 | +0.0092 | 0.447 | 0.000 |
| 60,000,000 | +0.0185 | +0.0085 | 0.542 | 0.225 |
| 70,000,032 | +0.0179 | +0.0089 | 0.487 | 0.000 |
| 74,000,016 | +0.0142 | +0.0124 | 0.597 | 0.147 |

A per-cycle turn-1–3 ratio of 0.000 is the noise correction clamping at zero: with 14 cells the
corrected sd(V) is not distinguishable from its own sampling noise. Read the pooled row.

## 8. Leave-one-opponent-out

Dropping any single bot leaves the bot-only slope in **+0.0093 … +0.0135** — dropping `random`,
the point with the most leverage, *raises* it to +0.0135. All nine variants stay positive.
Dropping any single opponent from the 14-opponent fit leaves it in **+0.0149 … +0.0219**. The
slope is not one point. (`sensitivity.json`; the per-variant CI is as wide as the main slope's,
so this is a leverage check, not nine independent tests.)

## 9. Hazards hit (a hazard is a finding)

1. 🚨 **`fit_ladder` silently returns an UNANCHORED ladder when called from the wrong cwd.**
   `elo.load_bot_anchors()` takes a RELATIVE default path (`data/gen3_bot_elo_anchors.json`), so a
   fit run from anywhere but a checkout root loses the bot pins and returns ratings near 1000
   instead of near 2000 with `anchored_to_bots: false` and no error. Mixing those with the bot
   anchors would have put the strength axis on two different scales. `refit_ladder.py` now chdirs
   to `repo_root()` and REFUSES a fit whose `anchored_to_bots` is false.
2. 🚨 **A cluster bootstrap draw has exactly as many entries as the original**, so a size test
   cannot tell a resampled selection from an unresampled one. A first revision passed a per-battle
   cell table where a per-selection one was needed and silently paired resampled battles with the
   ORIGINAL cells, scrambling battles across opponents; the tell was percentile CIs that did not
   contain their own point estimates. Caught by asserting exactly that.
3. 🚨 **A shadowed name mis-assigned the strength axis** in the opponent-resampled bootstrap
   (`picks` rebound to the per-battle cell codes, so each drawn cell got the strength of a
   different opponent). The tell was the same one: a bootstrap distribution centred well away from
   its point estimate. Both fixes are commented in `analyze.py`, and the bot-only opponent-resampled
   interval changed from "excludes zero" to "straddles zero" — i.e. the bug had been reporting a
   detection that is not there.
4. ⚠️ **A raw between-group variance is inflated by group size.** Comparing 216 teams against 14
   opponents on raw between-variance would have convicted the team on arithmetic alone (0.051 vs
   0.004). Both the §5 spread and the §6(d) contrast are noise- or null-corrected.
5. ⚠️ `eval_manifest.json`'s `opponent_pins` is `{}` on every cycle, so the sentinel→snapshot map
   had to come from `eval_results.jsonl` positionally. It is verified against the manifest's own
   win counts (all 20 cells match exactly) rather than trusted.

## 10. Ledger paragraph (for the orchestrator to append — this file does NOT edit the ledger)

> **2026-09-09 · MIXTURE DIAGNOSTIC on `ai_v12_02_winprob_critic` — DETECTED.** The win-prob
> critic emits close to one marginal win probability regardless of opponent. Over the last four
> trace cycles (50/60/70/74M; 29,495 states, 951 battles, 56 opponent-cycle cells, every cycle
> carrying a schema-1 manifest so nothing is SELECTION UNKNOWN, all statistics HT-reweighted by
> the capture rates), the between-opponent spread of `V` on turn-1–3 states is **0.334×
> [0.318, 0.451]** the between-opponent spread of the outcome — delta **−0.0731
> [−0.0840, −0.0588]**, battle-clustered, clear of zero — where for any calibrated critic the two
> must be EQUAL. Over all states the ratio is 0.573 [0.509, 0.689]. The bias `V` − true win rate
> runs **−0.092** vs `random` (true WR 1.000, V 0.908) to **+0.106** vs `sentinel_1` (0.635,
> 0.741), and regresses on opponent Elo at **+0.0171 [+0.0131, +0.0209] per 100 Elo**
> (14 opponents, cycle fixed effects; **+0.0094, +0.0300** with opponents resampled too). The
> bot-only slope is +0.0098 [+0.0059, +0.0127] with the nine pinned bots held fixed but
> **NOT DETECTED** [−0.0177, +0.0447] once the bots are themselves resampled — eight of nine sit
> within 127 Elo, so that stratum is underpowered by construction. Counter-hypotheses:
> loss-preferring capture ELIMINATED (reweighting SHRINKS the effect 6×; the raw bot slope is
> +0.0607); the greedy-sentinel handicap ELIMINATED and CONSERVATIVE (correcting it raises the
> slope to +0.0300 [+0.0261, +0.0338]); late-game dominance ELIMINATED (the bot slope is +0.0169
> early and −0.0095 late — strongest where the board says least). A per-TEAM effect SURVIVES as a
> SEPARATE finding: excess between-group residual variance is +0.0246 [+0.0202, +0.0385] for the
> trainee's team against +0.0027 [+0.0013, +0.0074] for the opponent, ~9×, so the critic fails to
> condition on its own team more badly than on the opponent — but teams are drawn independently
> of the opponent queue, so this adds noise to the slope rather than causing it. The head is not
> merely flat: at turn 1–3 its within-opponent spread is 0.0948 against a between-opponent 0.0229
> (4.1×). This is a quantified account of §4.2b's resolution failure (`sd_true_excess` 0.2550,
> low resolution in EVERY stratum): part of what the head is failing to resolve is simply *who it
> is playing*. Two tooling hazards recorded: `fit_ladder` silently returns an unanchored ladder
> from the wrong cwd, and a cluster-bootstrap draw is the same length as its source so a size test
> cannot detect an unresampled index (both cost a wrong interval before they were caught).
> Measurement: `designs/research_state/measurements/winprob_mixture_diagnostic_2026-09-09/`.

## 11. Proposed amendment to UNDERSTANDING.md §4.2b (text only — not applied here)

> Append to §4.2b, after the identity-test sentence:
>
> **The resolution failure has a named component: the head does not condition on the opponent.**
> [MEASURED, `winprob_mixture_diagnostic_2026-09-09`] Over the last four trace cycles the
> between-opponent spread of `V` on turn-1–3 states is 0.334× [0.318, 0.451] the between-opponent
> spread of the outcome, where a calibrated critic's must be 1.0; over all states, 0.573
> [0.509, 0.689]. Bias `V` − true win rate rises with opponent Elo at +0.0171 [+0.0131, +0.0209]
> per 100 Elo (14 opponents; +0.0094, +0.0300 with opponents resampled), running −0.092 against
> `random` to +0.106 against the strongest sentinel. The loss-preferring capture rate and the
> greedy-sentinel handicap both work AGAINST the finding, and the effect is largest on early
> states and reverses late. **A second conditioning failure, larger, was found in passing: the
> trainee's OWN team explains ~9× more of the critic's residual than the opponent does**
> (+0.0246 [+0.0202, +0.0385] vs +0.0027 [+0.0013, +0.0074], both in excess of a permutation
> null). **Open:** whether feeding the value path an opponent-strength scalar closes any of it —
> the D-ladder's conditioning arm, sketched in the measurement dir §12. **Caveat:** the bot-only
> slope is NOT DETECTED once the nine pinned bots are themselves resampled; the detection rests
> on the 14-opponent set and, more strongly, on the spread identity, which needs no strength axis.

## 12. Design sketch — the value-side opponent-conditioning arm (SKETCH ONLY, nothing built)

Add one route to the `_value_pooled_routes` seam in `extractor_forward.py`, alongside
`gen3_value_true_team_v1`: a small FiLM conditioner driven by a **two-number opponent code** —
the opponent's ladder rating, normalised (e.g. `(elo − 1800)/400`), plus a low-cardinality
opponent-CLASS embedding (scripted-bot / self-play-snapshot / unknown) — producing a `D_MODEL`
γ,β pair applied to `value_pooled`, with a zero-init output projection so the route is
bit-identical at init. It inherits the seam's whole contract for free, and that contract is the
reason to put it there: `ProjectionAssembler` returns `pi_combined` as a concat that does not
contain `value_pooled`, so `pi` is provably bit-identical at ANY weight and a policy change
cannot confound the critic result. Like the true-team route it must RAISE, never silently skip,
when built without its key. **The key difference from `gen3_value_true_team_v1`, and the reason
this one is a candidate to SHIP rather than a ceiling probe: the opponent's rating is not
privileged information.** At ladder play the target tier is a knob the operator sets, and in
training it is the pool's own bookkeeping — unlike the opponent's actual party, which exists only
inside the simulator. The measurement above also says what the first read should be: report the
turn-1–3 spread ratio (0.334 today) and the strength slope (+0.0171) as the arm's primary meters,
not strength, because a head that closes the ratio to 1.0 has fixed the thing this arm exists to
fix whether or not it ladders higher. Two cautions. First, **§6(d) says the larger conditioning
failure is on the trainee's OWN team**, so an opponent-only conditioner addresses the smaller
half; a `(own-team, opponent-code)` pair may be the better first arm, and the amortization-gap
line (`project_count_dominates_conditioning`) predicts the team half is the harder one. Second,
this is a value-side patch to a symptom — if the head is failing to resolve because the MC-only
target is too noisy, handing it the answer as an input may buy the meter without buying the
critic. The identity test (V against its own MC continuation) is what would tell those apart and
should be re-run on any arm that moves the ratio.
