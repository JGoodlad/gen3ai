# S′-HONEST — clustering addendum (2026-11-09): REPORTED, never decided

**The registered verdict is unchanged** (GO-HONEST, GOAL-B-REACHED; PREREG `e81f80657`, rule `searched_play`). This
addendum recomputes the read's three numbers from the banked cells under the dependence models the registered
analysis did not use, because an independent review asked whether the headline survives the project's own rule:

> **The team is the unit**: cluster-bootstrap over teams; state- and team-weighted statistics have disagreed in sign.
> (`../../../UNDERSTANDING.md` §7, rule 10)

**What it finds.**

1. **The search GAIN survives every dependence model.** G_U (GO-imputed) is +1.98 pp with a lower bound of +0.31 to
   +0.41 pp whether the unit is the deal, the matchup, the unit seed, our team, Kakuna's team or both teams at once.
   The paired difference has no team clustering (design effect 0.95 to 1.05).
2. **The 0.50 clearance of the searched SCORE does not.** 0.5145 clears 0.50 only when all 5,100 deals are treated
   as independent (LB 0.5008). Clustered on the matchup the LB is 0.4992; on the read's own unit seed 0.4976; on our
   team 0.4942; on Kakuna's team 0.4954; two-way 0.4908 (roles independent) or 0.4973 (a team one cluster in either
   role). "Goal (b) met with search" is therefore a statement about these 216 teams and these deals under
   independence, as the PREREG disclosed. As a claim about unfamiliar teams in general it is NOT established: the
   point is above 0.50 and the one-sided mass at or below 0.50 is 0.05 to 0.12.
3. **The gain is carried by P72** (+4.62 pp; H +1.32 and F 0.00 both span 0; H and F together +0.66 [−1.39, +2.67]).
   The PREREG predicted no panel ordering (§13), so the contrast is post hoc: P72 − F is +4.62 pp [+0.65, +8.58],
   P72 − H spans 0, and the omnibus test reads p = 0.04 to 0.07 before any correction for looking. The gain is
   detected on the procedural panel only; it is not shown absent on the other two.
4. **P72's generator and the search's hidden-team sampler read the same Smogon tables** (confirmed at source, same
   git blobs). The simple mechanism "the sampler guesses P72's teams better" is REFUTED by a measured covariate: the
   sampled worlds are slightly FARTHER from the truth on P72 than on H or F. What remains is untested.

## 1. What the registered analysis did, against rule 10

| statistic | registered interval | its unit | follows rule 10? |
|---|---|---|---|
| G_U (decides GO-HONEST) | percentile bootstrap, B = 2,000 | the (panel, unit seed): 85 a panel, 20 deals each | no: a unit seed is not a team. Harmless here (finding 1) |
| searched U score (decides GOAL-B) | Wilson at the effective n | the deal, all 5,100 independent | no. DISCLOSED in PREREG §5 and the README ("a claim about these panels' teams, not a population beyond them"; review item m9) |

So the read did not break its registration; it registered a fixed-panel claim and said so. Two things the disclosure
did not cover: the two decided statistics used DIFFERENT units (had GOAL-B used the gain's own unit-seed bootstrap,
its LB would be 0.4976), and even with the teams held fixed the deals are not independent (§2).

## 2. The design the clusters come from (read from the cells)

- Each panel: 72 teams, every one piloted by us and by Kakuna. 1,700 deals a panel = 85 unit seeds × 20 games.
- **FINDING: games g and g + 10 of a unit seed play the SAME two teams** (the two challenge halves reuse the draw):
  850 matchup draws a panel, each played twice an arm, 779 distinct (ours, theirs) cells, 20 mirror deals.
  `clustering.py` asserts it on every deal. The PREREG's "every game's two teams are independent draws" (§3) holds
  across matchups, not across games, so the i.i.d. interval is too narrow even for the fixed-panel claim; the
  `matchup` row is the fixed-panel-correct one.
- **FINDING: our team sequence at unit seed s + 1 is Kakuna's at seed s**, so adjacent unit seeds are not independent
  draws of teams either.
- **FINDING: `record.json`'s game tuples carry our team and not the opponent's.** `extract_deals.py` joins each deal
  to the unit's `peer_*_report.json` (`team_draws`) and refuses when the two arms disagree on either team.

## 3. The dependence models, and why two-way is not a bootstrap

Panels are fixed strata (never resampled); a pooled replicate is the mean of the three panels' replicates. Every
statistic is a ratio sum(num) / sum(den) over a panel's deals, so a cluster replicate re-forms the ratio.

| model | cluster | what it answers |
|---|---|---|
| `iid` | the deal | the registered reading |
| `matchup` | (unit seed, game mod 10), 850 a panel | these teams, new pairings and dice: the fixed-panel claim done right |
| `unit_seed` | (panel, unit seed), 85 | the registered unit of the gain |
| `our_team` | the team we pilot, 72 | rule 10 on our side (the review's scheme) |
| `their_team` | the team Kakuna pilots, 72 | rule 10 on the opponent's side |
| `two_way` | ours × theirs crossed, roles independent | Cameron, Gelbach & Miller (2011): V_ours + V_theirs − V_cell |
| `dyadic` | a team is ONE cluster in either role | Aronow, Samii & Assenova (2015): two deals covary iff they share a team |
| `pigeonhole` | both roles resampled, weight c_ours × c_theirs | Owen (2007); a known-conservative bracket |

One-way models are a percentile cluster bootstrap (B = 20,000, `SeedSequence([20261109, model, panel])`) checked
against the analytic cluster-robust sandwich; they agree to 0.02 pp (`tables.md`), which licenses the sandwich for
the two crossed models. **Why not a naive two-way bootstrap:** no bootstrap is unbiased for crossed random effects
(McCullagh 2000); the pigeonhole bootstrap's variance is about V_ours + V_theirs + 3 V_noise, so it triple-counts
the game noise that dominates here (design effect 5.5 against CGM's 3.0). Removing 2 × the i.i.d. variance returns
it to CGM's (1.31 vs 1.21 pp on the score, 0.85 vs 0.82 pp on the gain). It is `design_beat_metamon_path.md` §8
item 7's "2.02× the binomial width" artefact, and it is reported only as the outer bracket.

**CGM against dyadic.** Both sides draw from the same 72 teams, so a new sample of teams would replace a team in
BOTH roles. A strong team lifts our score when we pilot it and lowers it when Kakuna does; the dyadic model counts
that negative cross-role covariance and CGM does not. Dyadic is the design-faithful team-as-sample interval; CGM is
its conservative companion (as if the two sides' teams were sampled separately). Neither clears 0.50.

## 4. Results (`tables.md` is GENERATED and complete; `clustering.json` holds every cell)

The script first reproduces the record to 1e-9 (the three registered gain intervals, the Wilson interval, the win
counts 867 / 895 / 862) and refuses otherwise.

| dependence model | searched U score | P(≤ 0.50) | raw U score | gain G_U, pp (GO-imputed) | P(≤ 0) |
|---|---|---|---|---|---|
| REGISTERED | 0.5145 [0.5008, 0.5282] | | 0.4948 | +1.98 [+0.32, +3.47] | |
| i.i.d. deals | 0.5145 [0.5008, 0.5280] | 0.018 | 0.4948 [0.4816, 0.5085] | +1.98 [+0.35, +3.57] | 0.009 |
| matchup draw | 0.5145 [0.4992, 0.5292] | 0.032 | 0.4948 [0.4798, 0.5097] | +1.98 [+0.33, +3.64] | 0.009 |
| unit seed | 0.5145 [0.4976, 0.5312] | 0.046 | 0.4948 [0.4797, 0.5099] | +1.98 [+0.41, +3.56] | 0.007 |
| **our team** | 0.5145 **[0.4942, 0.5353]** | 0.082 | 0.4948 [0.4742, 0.5156] | +1.98 **[+0.31, +3.63]** | 0.010 |
| **their team** | 0.5145 **[0.4954, 0.5337]** | 0.069 | 0.4948 [0.4765, 0.5135] | +1.98 **[+0.39, +3.56]** | 0.007 |
| **two-way (CGM)** | 0.5145 **[0.4908, 0.5382]** | 0.115 | 0.4948 [0.4715, 0.5181] | +1.98 **[+0.38, +3.58]** | 0.008 |
| **two-way (dyadic)** | 0.5145 **[0.4973, 0.5317]** | 0.049 | 0.4948 [0.4765, 0.5131] | +1.98 **[+0.38, +3.58]** | 0.008 |
| pigeonhole (bracket) | 0.5145 [0.4827, 0.5469] | 0.185 | 0.4948 [0.4631, 0.5263] | +1.98 [−0.85, +4.84] | 0.083 |

The paired-only gain (+2.00 pp) and the NULL-imputed one (+2.04 pp) move the same way; the NULL-imputed upper bound
stays above +3 in every model, so `go_below_margin` stays false.

**The gain by panel (GO-imputed, pp).** The paired-only P72 point is the record's +4.69.

| dependence model | H | F | P72 | H + F (P72 left out) |
|---|---|---|---|---|
| unit seed (registered unit) | +1.32 [−1.53, +4.15] | 0.00 [−2.74, +2.79] | +4.62 [+2.00, +7.24] | +0.66 [−1.31, +2.65] |
| our team | +1.32 [−1.70, +4.25] | 0.00 [−2.76, +2.66] | +4.62 [+1.78, +7.52] | +0.66 [−1.39, +2.67] |
| their team | +1.32 [−1.61, +4.32] | 0.00 [−2.56, +2.52] | +4.62 [+1.95, +7.29] | +0.66 [−1.28, +2.66] |
| two-way (CGM) | +1.32 [−1.84, +4.48] | 0.00 [−2.41, +2.41] | +4.62 [+1.91, +7.33] | +0.66 [−1.32, +2.65] |
| two-way (dyadic) | +1.32 [−1.86, +4.51] | 0.00 [−2.11, +2.11] | +4.62 [+1.71, +7.52] | +0.66 [−1.25, +2.57] |

**The score by panel.** No panel's searched score clears 0.50 once clustered on a team: H 0.5100 [0.4744, 0.5467],
F 0.5265 [0.4936, 0.5583], P72 0.5071 [0.4697, 0.5450] (our team). Raw P\*: H 0.4962, F 0.5274, P72 0.4609.

**The panels against each other (POST HOC, unadjusted for three contrasts; our-team variances).** P72 − F +4.62 pp
[+0.65, +8.58] (p = 0.022); P72 − H +3.29 [−0.86, +7.45] (p = 0.12); P72 − mean(H, F) +3.96 [+0.44, +7.47]
(p = 0.027); omnibus Q = 5.44 on 2 df, p = 0.066 (0.041 to 0.066 across the models). With a Bonferroni factor of 3
the largest contrast is p ≈ 0.07.

## 5. P72 and the sampler: the same tables, checked at source

- **P72's generator** is `src/rust_sim/harness/ou_random_teams.js` (`--emit 200 --seed 3721`, frozen at `cb0f1769`):
  the lead in proportion to usage, each mate in proportion to the lead's teammate table, every set fact (moves,
  item, ability, spread, Hidden Power) an independent draw from that species' marginals.
- **The honest search's hidden-team sampler** is `src/main/value_search/world_draws.py` (`Prior`, through
  `agents.gen3_data.priors`): a never-seen species in proportion to its mean teammate co-occurrence with the
  revealed ones, its set from the same marginals, screened against what the replay has shown.
- **They read the same seven files under `data/pokemon/`, and the blobs are IDENTICAL at the generator's freeze
  (`cb0f1769`) and at the read's pin (`e46e3ddf8`)**: `gen3_smogon_stats` `98b25fc9fc52`, `gen3_move_usage`
  `a16bff901f4b`, `gen3_item_usage` `f2e3e94099a0`, `gen3_ability_usage` `bda5fd6826af`, `gen3_spread_usage`
  `af67ac08ed55`, `gen3_teammate_usage` `a12fff17b34b`, `gen3_hidden_power_usage` `d82e2d6c3c1a`
  (`git rev-parse <commit>:data/pokemon/<file>.json`).
- **It is not an information leak.** The sampler reads public priors and our side's own observations; the read's
  leak test passes on every channel. The prior is CORRECTLY SPECIFIED for P72 by construction, which no panel of
  human-built teams can be.
- **H is not fully human either:** `heldout72_hpfix` is replay-derived, and the fields a replay never revealed are
  filled from the same usage tables (X61's REVIEW B1). The ordering F 0.00 < H +1.3 < P72 +4.6 matches "the share
  of the opponent's team drawn from usage"; nothing here establishes it.

**The covariate that tests the simple mechanism** (`prior_match.py`; counted searched games, means over a game's
searched decisions and worlds, interval = bootstrap over the opponent's true team):

| read | panel | games | species hit | at the first searched decision | move hit | item hit |
|---|---|---|---|---|---|---|
| HONEST | H | 1,695 | 0.862 [0.854, 0.869] | 0.668 | 0.686 [0.676, 0.696] | 0.948 [0.940, 0.957] |
| HONEST | F | 1,681 | 0.866 [0.857, 0.874] | 0.670 | 0.696 [0.683, 0.709] | 0.941 [0.925, 0.955] |
| HONEST | P72 | 1,696 | 0.842 [0.833, 0.850] | 0.642 | 0.676 [0.666, 0.686] | 0.936 [0.924, 0.947] |
| LEAKY (S′-LIVE) | H | 900 | 0.861 [0.851, 0.870] | 0.671 | n/a | n/a |
| LEAKY (S′-LIVE) | F | 895 | 0.874 [0.864, 0.883] | 0.682 | n/a | n/a |
| LEAKY (S′-LIVE) | P72 | 900 | 0.836 [0.826, 0.846] | 0.634 | n/a | n/a |

The worlds hold FEWER of the opponent's true species on P72, and no more of its moves or items, in both reads. So
"the sampler reconstructs P72's teams better" is refuted in its hit-rate form. Two more facts against a stable
P72-specific effect of the prior: the leak-inclusive S′-LIVE, with the same team sampler, gained no more on P72
(+4.11) than on H (+4.53) or F (+5.15); and the departure rate is 0.371 to 0.373 on every panel.

**What is left, none of it tested:** (i) calibration: a world drawn from P72's own generative law can be a better
basis for averaging even when it matches no more facts; (ii) headroom: raw P\* is weakest on P72 (0.461 against
0.496 and 0.527), and the searched scores end level (0.507, 0.510, 0.527); (iii) exclusions: F loses 10.0 % of its
searched decisions (H 2.6 %, P72 3.3 %); (iv) chance: one of three post-hoc contrasts at p ≈ 0.02.

**JUDGEMENT.** The P72 concentration is a caveat, not a demonstrated artefact. What the data support: the gain is
not shown on human-built teams (F 0.00 [−2.8, +2.7]; H + F +0.66 [−1.4, +2.7]), and P72 is the one panel where the
search's prior is the truth. A deployment against human teams should expect the F and H figures until a read on
human-built teams says otherwise.

## 6. What may be quoted

- "Leak-honest search beats P\* alone against Kakuna on U by +1.98 pp; the lower bound is above 0 under every
  clustering (+0.31 pp at worst)." With its caveat: detected on P72 only.
- "The searched arm scores 0.5145 on U; registered interval [0.5008, 0.5282] (GOAL-B-REACHED as registered);
  team-clustered [0.494, 0.535], two-way [0.491, 0.538] or [0.497, 0.532]." Never "goal (b) met" without both.
- Not: "search does not work on human teams" (H + F cannot exclude +2.7 pp); not: "the P72 gain is a leak".

## 7. FINDINGs

1. Deals are not independent even within the fixed panel: the g / g + 10 matchup reuse (§2). A future
   `searched_play` registration should state its unit for the score and use the same one for both statistics.
2. Team sequences are shared across adjacent unit seeds in opposite roles (§2).
3. The panel contrasts are post hoc and unadjusted; the PREREG predicted none (§13).
4. `move_hit` and `item_hit` mix what play revealed with what the prior drew (a revealed fact is true in every
   world); they are comparable ACROSS panels only. S′-LIVE's rows carry no world teams, hence n/a.
5. NOT verified: whether the sampler is better CALIBRATED on P72 (no proper scoring rule was run); whether the
   pattern replicates (one read, one seed range, one opponent checkpoint).
6. The crossed intervals are analytic (normal), on 72 clusters a panel; the dyadic estimator's small-sample
   behaviour is not checked beyond the G / (G − 1) correction.
7. `prior_match.py` reads the archive (`~/gen3ai_archive/…`), which is not in the repository; its two JSON outputs
   are committed.

## 8. Reproduce

```bash
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
"$PY" extract_deals.py ~/gen3ai_archive/sprime_honest_play_2026-11-06 > deals.csv
"$PY" clustering.py deals.csv clustering.json tables.md
"$PY" prior_match.py ~/gen3ai_archive/sprime_honest_play_2026-11-06 prior_match_honest.json
"$PY" prior_match.py ~/gen3ai_archive/sprime_live_play_2026-11-01 prior_match_leaky.json
```

Files: `extract_deals.py`, `deals.csv` (5,100 rows), `clustering.py`, `clustering.json`, `tables.md`,
`prior_match.py`, `prior_match_honest.json`, `prior_match_leaky.json`. Frozen record scripts, never tools.
