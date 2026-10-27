# Piloting-gap forensics: where P\* loses the extra unfamiliar-team games against Kakuna (2026-10-26)

**Adversarial review (2026-10-27): [`../h_mid_value_test_2026-10-26/REVIEW.md`](../h_mid_value_test_2026-10-26/REVIEW.md).** Answer 3 ("it comes with the TEAM, not the set") is not evidenced: KOs per entry restate the result, and the within-team comparison cannot separate team from set (B1). The phase picture is corroborated by V (m3).

**Question.** P\* (`rb_x26_s2601` @280,000,170) beats Kakuna ckpt 34 **0.598** on home72 and **0.487** on the
unfamiliar panels (X51). Where in the game do the extra losses come from, and what kind of decision carries them?
Banked games only: no new play, no model run, CPU, post hoc and DESCRIPTIVE (nothing here was registered).

Abbreviations: pp = percentage points; U = H (`heldout72_hpfix`), F (`forum72`) and P72 with equal weight (X51's
definition); "our" = the model's side; D(t) = the material lead at the start of turn t (our summed HP fractions over six
mons minus theirs; a mon that never entered counts 1.0); DiD = difference in differences. Every interval is a 95 %
percentile bootstrap over 20-game units, resampled within each cell (B = 1,000).

## Answer

1. **The gap is not opened in the opening.** Through turn 10 the material state carries none of the 11.2 pp gap
   (state-at-turn-10 part −0.4 pp [−2.6, +1.3]); P\* loses its first mon equally often on both sets (first faint is
   Kakuna's 0.475 vs 0.474). The gap accrues from materially LEVEL positions in the mid and late game: from level
   material at turn 10 (75–78 % of games) P\* wins **0.609 [0.559, 0.664] on home72 and 0.501 [0.464, 0.541] on U
   (+10.8 pp [+4.8, +17.0])**. By turn 20 the state carries 2.1 pp, by 30 5.8, by 40 8.1 [3.3, 12.8] of the 11.2.
2. **What P\* does differs by team set, not by pilot.** The mix of decision types (attack / switch / setup / status /
   hazard / recovery / protect / phaze / Explosion) shifts between home72 and U, and Kakuna's mix shifts the same way
   in the same games. Relative to Kakuna no class moves by more than ±1 pp (all but one interval cover 0, and that one,
   phaze / haze −1.0 pp, does not replicate). No Explosion measure moves detectably. The gap is in HOW WELL each
   decision lands.
3. **The premium lives in P\*'s own mons, and it comes with the TEAM, not the set.** Per mon that entered, P\*'s mons
   out-kill Kakuna's by **+0.081 [+0.017, +0.144] on home72 and −0.028 [−0.069, +0.013] on U**. Counting only mons
   whose exact set (species + moves) is in our pool, on both sides, U still reads **−0.012 [−0.067, +0.047]**: a
   familiar set on an unfamiliar team plays like Kakuna's. Δ home72 − U on seen sets **+0.093 [+0.004, +0.176]**,
   against a further −0.03 for unseen sets. The premium is concentrated on COMMON species (pool share > 10 %:
   +0.090 → −0.022), so it is not about rare mons.
4. **Facing is the smaller channel, as X37a said.** Kakuna's mons with sets our pool never had kill no more against
   P\* than its seen-set mons (0.658 vs 0.675 per entry), and its species rare in our pool kill LESS (0.59). The one
   facing-like measure that moves: P\*'s edge in hitting Kakuna's switch-ins super-effectively (−1.9 pp on home72)
   is gone on U (DiD −1.7 pp [−3.4, +0.0]; against PastaMittens −2.7 [−4.7, −0.6]), and so, less clearly, is its
   edge in super-effective hits (+2.7 [−1.4, +6.8]; PastaMittens +4.4 [+0.6, +8.1]).
5. **When only our side pilots an unfamiliar team (X37a, X26 @161M), the deficit opens earlier.** Piloting H against
   Kakuna on home72, X26 trails in material from turn 3 (D(10) −0.19 vs +0.07 when it faces H instead), and against
   Kakuna piloting the SAME H teams it trails by D(20) −0.29 [−0.49, −0.08] more. The whole pilot-swap difference
   (4.75 pp) is in the material state by turn 15–20.

**Replication.** P\* against PastaMittens (1,600 twin games: the same teams, seeds and halves) shows each pattern:
state at 10 carries −0.7 pp of a 6.5 pp gap, level-at-10 conversion +7.3 pp [+0.3, +14.7], the seen-set per-mon
edge +0.072 [−0.021, +0.167]. The twins share our teams and our deterministic policy, so this replicates the
opponent, not the team sample.

## 1. Game phase (`results.json` `decomposition`, `material`, `conditional`)

The gap W_home − W_U splits exactly into a STATE part (how often each side of the comparison is ahead / level /
behind at turn t, at the average conversion) and an AFTER part (conversion from the same state). Seven bins of D(t),
edges ±0.4, ±1, ±2 mons; a game already over is read at its final state.

| t | P\* vs Kakuna: state part | after part | share in state | P\* vs PM: state | X26 face-H − pilot-H: state | of gap |
|---:|---|---|---:|---|---|---:|
| 1 | 0 | 11.2 [5.0, 16.8] | 0 % | 0 | 0 | 25.3 |
| 5 | −0.3 [−1.3, +0.7] | 11.5 | −3 % | −0.4 | +0.9 [−0.2, +2.4] | 25.3 |
| 10 | −0.4 [−2.6, +1.3] | 11.6 [6.5, 16.7] | −3 % | −0.7 | +3.7 [+1.8, +5.7] | 25.3 |
| 15 | +1.2 [−1.6, +4.0] | 10.0 | 11 % | +1.3 | +7.1 [+4.9, +9.5] | 25.3 |
| 20 | +2.1 [−1.4, +5.6] | 9.1 [4.6, 13.5] | 19 % | +1.8 | +12.1 [+8.5, +15.8] | 25.3 |
| 30 | +5.8 [+1.3, +10.1] | 5.4 | 52 % | +3.2 | +15.2 [+9.9, +20.1] | 25.3 |
| 40 | +8.1 [+3.3, +12.8] | 3.1 [0.0, 6.0] | 72 % | +5.9 | +20.2 [+14.5, +26.2] | 25.3 |

- **Conversion from the same material** (P\* vs Kakuna, home72 / U): level at turn 10 0.609 / 0.501 (+10.8
  [+4.8, +17.0]); ahead at 10 0.851 / 0.685 (+16.6 [+5.9, +27.6]); level at 20 0.630 / 0.487 (+14.3 [+6.1, +22.0]);
  after losing the first mon 0.543 / 0.390 (+15.3 [+6.7, +24.3]); after taking the first KO 0.661 / 0.594 (+6.7
  [+0.1, +12.8]). The shares of games ahead / level / behind at 10 and 20 do not differ (each |Δ| ≤ 3.2 pp, every
  interval covers 0).
- **U games run 9.2 turns longer** (47.6 vs 38.3 mean, [−12.8, −6.0]), so a fixed turn sits earlier in a U game. Even
  allowing for that, nothing separates the sets before turn ≈ 15.
- **Turn 1.** The lead's first action differs by set (on home72 15.5 % status, on U 6.0 %, more switching and
  hazards), identically against PastaMittens (greedy play is deterministic). It buys nothing measurable: D(3) and D(5)
  are equal on both sets.
- **X37a's reverse cell** (X26 pilots H, Kakuna home72) trails early: D(3) −0.07, D(10) −0.19 [−0.28, −0.11],
  D(20) −0.42, first faint ours 58 %. Kakuna piloting the same H teams against itself (`kcal`) trails D(20) −0.13; the
  difference (+0.29 [+0.08, +0.49]) puts the whole 4.75 pp pilot gap into the state by turn 15–20. These are X26
  @161M games, a different checkpoint from P\*, and the reverse cell has no P\* twin.

## 2. Decision types (`decisions`, `decisions_did`, `turn1`)

Each side's normal (not forced) decisions, classed by the move's dex data (`agents.gen3_data.moves`): attack, switch,
setup, status, hazard, recovery, protect / Substitute, phaze / Haze, Explosion / Self-Destruct, other. In a symmetric
cell both sides' teams come from the same panel, so a team-set effect moves both sides alike and the DiD (P\* minus
Kakuna, home72 minus U) isolates the pilot.

| measure (per game) | P\* − Kakuna, home72 | U | DiD | DiD vs PastaMittens |
|---|---|---|---|---|
| attack share | +3.2 | +2.4 | +0.8 [−2.5, +4.5] | +1.9 [−1.3, +5.0] |
| switch share | −1.8 | −0.7 | −1.1 [−2.7, +0.8] | −1.0 [−2.4, +0.6] |
| setup share | +1.1 | +0.1 | +0.9 [−0.2, +2.2] | +0.7 [−0.8, +2.1] |
| status / hazard / recovery / protect shares | | | each within ±0.6 | each within ±0.7 |
| phaze / haze share | −1.0 | 0.0 | −1.0 [−1.6, −0.4] | −0.3 [−1.0, +0.3] |
| Explosion used (games) | +6.5 | +3.2 | +3.3 [−5.4, +12.1] | +3.7 [−2.7, +10.8] |
| Explosion wasted (immune / Protect / miss) | +1.2 | +4.3 | −3.2 [−10.0, +4.1] | −0.3 [−6.8, +6.0] |
| attacks super-effective | +1.4 | −1.3 | +2.7 [−1.4, +6.8] | +4.4 [+0.6, +8.1] |
| status moves failed | −1.1 | +0.3 | −1.5 [−3.3, +0.2] | −0.3 [−2.2, +1.7] |
| switch-ins hit super-effectively or KO'd that turn (ours − theirs) | −1.9 | −0.1 | −1.7 [−3.4, +0.0] | −2.7 [−4.7, −0.6] |
| KOs per game | +0.41 | −0.14 | +0.56 [+0.19, +0.93] | +0.48 [+0.08, +0.85] |

(pp unless stated.) Kakuna's own rates shift with ours: its recovery share rises 2.0 pp on U, ours 1.5; both sides
explode in fewer U games (P\* 38 → 26 %, Kakuna 32 → 23 %). The two pilot-specific movers that replicate against
PastaMittens are both about TARGETING: P\*'s super-effective share and its punishment of the opponent's switch-ins
fall on U relative to Kakuna's. Both point the same way against both opponents and exclude 0 against PastaMittens;
against Kakuna their intervals touch or cover 0. With 20+ measures read, one interval excluding 0 is expected by
chance, so these are leads, not findings.

## 3. Matchup knowledge (`mons`)

Per mon that entered, its KOs and whether it fainted, bucketed by whether its set is in our 719-team pool (mon tier:
species + moves, Hidden Power typed; `main.anchors.team_overlap`) and by its species' pool share (the fraction of
pool teams carrying it).

| P\* vs Kakuna, U | P\*'s mons: KOs per entry | Kakuna's mons |
|---|---|---|
| set in pool | 0.663 [0.626, 0.697] (n 3,232) | 0.675 [0.639, 0.710] (3,192) |
| set not in pool | 0.616 [0.587, 0.649] (3,723) | 0.658 [0.626, 0.691] (3,731) |
| species pool share > 10 % | 0.657 [0.630, 0.685] | 0.679 [0.655, 0.702] |
| 2–10 % | 0.671 [0.598, 0.752] | 0.653 [0.557, 0.754] |
| ≤ 2 % | 0.453 [0.392, 0.513] (559) | 0.589 [0.517, 0.660] (548) |
| absent | 0.40 [0.25, 0.56] (100) | 0.39 [0.26, 0.53] (94) |

- **Threats P\* rarely sees do not beat it more.** Kakuna's rare-species and unseen-set mons kill no more than its
  common, seen ones. If P\* misplayed specific unfamiliar threats, those mons would out-kill the rest.
- **P\* pilots rare species badly, on any team.** Its ≤ 2 % species kill 0.45 per entry on U against Kakuna's 0.59 in
  the same games (P\* − Kakuna −0.137 [−0.224, −0.046]; against PastaMittens −0.150 [−0.250, −0.046]), and 0.43 vs 0.64
  on home72 (n = 68 / 70, −0.22 [−0.44, +0.02]). X26 piloting H: 0.30 vs Kakuna's 0.41 on the same teams. These mons
  are 9 % of P\*'s entries on U, so at most ≈ 0.08 of the 0.56 KOs-per-game DiD (≈ 15 %).
- **Most of the premium is on common species:** P\* − Kakuna per entry at pool share > 10 %: home72 +0.090 [+0.020,
  +0.153], U −0.022 [−0.067, +0.022].

## 4. Our team's role vs theirs (`mons_relative`, `familiarity_regression`)

On U both sides pilot unfamiliar teams, and the panels mix familiar and unfamiliar sets on both sides. Two reads
separate them.

**Per mon, P\*'s KOs per entry minus Kakuna's** (both sides restricted alike):

| | home72 | U | Δ |
|---|---|---|---|
| all mons | +0.081 [+0.017, +0.144] | −0.028 [−0.069, +0.013] | +0.109 [+0.034, +0.188] |
| sets in our pool only (both sides) | +0.081 (all are) | −0.012 [−0.067, +0.047] | +0.093 [+0.004, +0.176] |
| sets not in our pool only | — | −0.042 [−0.097, +0.013] | |
| PastaMittens: all / seen only | +0.065 / +0.065 | −0.033 / −0.007 | +0.098 [+0.021, +0.177] / +0.072 [−0.021, +0.167] |

A familiar set loses most of P\*'s edge as soon as it sits on an unfamiliar team; unfamiliarity of the set itself
costs about a third as much again. X37a's pilot swap agrees: on the same H teams against the same opponent, X26's
seen-set mons kill like Kakuna's (0.591 vs 0.595 per entry), its unseen-set mons less (0.505 vs 0.551).

**Within U, win probability against the number of each side's sets found in our pool** (logistic, panel fixed
effects, n = 1,200): **+3.0 pp [+0.5, +5.4] per OUR seen set**, **−1.9 pp [−4.9, +0.9] per THEIR seen set**
(PastaMittens +3.4 [+1.2, +5.7] / −1.8 [−4.3, +1.1]). A seen set is also a standard set, so each coefficient mixes
the set's strength (which helps whoever holds it) with familiarity. If strength helps both holders alike, the sum,
**+1.0 pp [−3.3, +5.2] per set**, is the familiarity part (piloting plus facing, at set level): small, not
detected. A team-context covariate (the most of our sets shared with any ONE pool team; U never exceeds 3) is too
collinear with the set count to read (+1.9 pp [−4.7, +8.0]).

**Reading.** Of P\*'s +0.11 KOs-per-mon home-team edge, about 0.09 goes with the team context and about 0.03 with
set novelty. Two caveats: on home72 Kakuna's mons are also familiar to P\*, so the context part includes some facing
(X37a puts facing at ≈ 1.7 of ≈ 14 pp); and "set in pool" is the mon tier (item and spread are not matched).

## Hypotheses: what training lever would close the gap

Ranked by how much of the gap each could reach. Each has a falsifiable test.

**H-CTX (team-context memorisation; the largest).** P\* learned how to play its 719 particular SIXES (role
assignments, which mon to preserve, win conditions), not portable set or species knowledge. A familiar set on a new
team gets none of it, and the loss shows up from turn 15 on, where a team plan matters. Lever: team-level breadth
whose PARTS are familiar: recombine pool sets into new teams, the complement of D3 (new sets inside old teams).
- **Test (CPU, ≈ 1.5 lane-h, no training):** a recombined panel R72, 72 teams each built from six mon sets that are
  in the pool, no two of them on the same pool team, under Smogon teammate usage so the teams stay coherent. Play P\*
  vs Kakuna symmetric, seeds 0–19 (400 games), plus a Kakuna-vs-Kakuna R72-vs-home72 cell to rate R72's strength.
  H-CTX predicts R72 ≈ U (≤ 0.52) and a seen-set per-mon edge ≈ 0; the set-level account predicts R72 ≈ home72
  (≥ 0.57). Kill rule: R72 ≥ 0.57 kills H-CTX.

**H-MID (mid-game conversion).** The extra losses come from level positions after turn 10. Either the value function
misjudges unfamiliar mid-game positions, or the policy cannot act on a correct judgement. The two need different
levers (value / search vs policy breadth).
- **Test (CPU, forward passes only):** re-encode P\*'s turn-10 and turn-20 states from the banked captures (the START
  command, seed and every choice are stored; play is deterministic) and read V. Compare its calibration on level
  states, home72 vs U. If V is calibrated on U (it expects to lose those level positions), the deficit is the
  policy's, and a one-ply search on V has headroom on U. If V is optimistic only on U, the critic shares the
  memorisation and search cannot help. Prediction: V less calibrated on U (Brier worse by ≥ 0.02).

**H-RARE (rare-species piloting).** P\*'s mons of species on ≤ 2 % of pool teams kill ≈ 0.14 per entry less than
Kakuna's, on any team. Upper bound ≈ 15 % of the gap. Lever: species-balanced team sampling (upweight teams carrying
rare species), or a corpus sampled by Smogon usage (N1-P's procedural teams do this).
- **Test:** on N1-P's or D3's banked Kakuna games, parsed with this unit's scripts, the ≤ 2 % bucket's per-entry
  deficit against Kakuna halves (to ≥ −0.07) while the > 10 % bucket holds. No change kills H-RARE as a lever.

**H-TGT (opponent-switch anticipation).** The facing share: P\* punishes the opponent's switch-ins and lands
super-effective hits less often relative to Kakuna on U (both point the same way against both opponents; each
interval excludes 0 only against PastaMittens). Anticipating a
switch-in needs a model of the opponent's team, and D0f showed the belief deltas are not that model in games. The
policy's own memory of pool opponents is. Lever: opponent-side breadth (`--team-mix-sides both`, already in D1 / D3 /
N1-P).
- **Test:** the switch-punish DiD on any breadth arm's banked games moves toward 0 by ≥ 1 pp against C0. No move
  means this channel is not the breadth lever's.

## How these bear on the work in flight

- **D3 (set mutation, X41):** it keeps each pool team's species core and redraws 1–2 sets, so it trains the
  set-novelty part (≈ 0.03 of 0.11 KOs per mon) and leaves the team context familiar. Under H-CTX its ceiling is
  ≈ a quarter of the premium (≈ 3 pp of U), consistent with look 1's +1.3 pp. A D3 gain should show in unseen-set
  mons first.
- **N1-P (12,000 procedural teams, both sides):** new team contexts at 10⁴. That is H-CTX's lever, if contexts learnt
  from procedural teams transfer to human-built ones (F is the clean panel). Procedural teams may lack coherent team
  plans, the thing H-CTX says is memorised. The per-mon seen-set edge on F is the mechanism read to add beside Δ_F.
- **X59-S (matched students: human vs self-play states):** human replays carry many human team contexts. Under H-CTX
  the human-state arm's value, if any, should show on U more than on home72. A student read on home72 alone would
  miss it.
- **N2 (the exact-set identity test, registered `98bcb1146`):** it asks whether P\* keys on the exact identity of its
  own sets. This read predicts COMPOSITIONAL at set level (a familiar set on a new team plays like Kakuna's, an
  unfamiliar one only ≈ 0.03 KOs per entry worse), with any identity effect at TEAM level, which N2's team-level
  guard is placed to see.
- **Belief levers (D7, lever 1 (b) / (c)):** nothing here revives them. Facing is the minor channel, and its one
  marker (switch-in anticipation) is policy targeting, not belief calibration.

## FINDINGS (standing rule 7)

1. **No value (V) traces exist for anchor games.** Captures hold the protocol, both teams, the seed and every choice,
   but not our model's per-decision outputs. "When does the win-probability swing" is read through material, a crude
   state proxy. A position can be level in HP and lost in matchup, so the state part UNDER-states how early the gap
   forms. H-MID's test is the V read.
2. **The X26 @161M cells are a different checkpoint.** §1's reverse / `kcal` contrasts are X26 @161M (X37a), not P\*.
   No P\* reverse cell was ever played (X51 FINDING 4).
3. **The PastaMittens replication is not independent of team sampling** (twins share teams, seeds and our
   deterministic policy).
4. **Kakuna's peer aliases do not name the team sets.** In X37a's `kcal` cell, `MetaKakunaAway*` pilots home72 and
   `MetaKakunaHome*` pilots H (checked by matching each side's six to the panel files). This unit's focal side is
   the H pilot.
5. **20 games stopped by the 250-turn forfeit** leave each side's last request unanswered; the parser drops that one
   decision. All 5,400 captures parsed, and every protocol result equals its `games.jsonl` row.
6. **Many measures, few replicated.** §2 reads 20+ DiDs; only the two targeting measures point the same way against
   both opponents (each excludes 0 against PastaMittens only).
   No multiplicity correction is applied. Treat a single-opponent interval excluding 0 as a lead, not a finding.
7. **The phase decomposition uses absolute turns** while U games run 9 turns longer. A per-game-fraction version was
   not computed.

## Files

- `scripts/parse_games.py`: captures → one row per game (both sides' decisions, every move's outcome flags, faints,
  a material snapshot each turn). Reads `/home/goodlad/dev/gen3ai-reads/{x51_2026-10-20, mm_strongest_2026-10-25,
  x37a_2026-10-18}/cells/*/capture/`: 5,400 games (X51 1,600; PastaMittens 1,600; X37a deploy 600, reverse 400,
  forum72 400, p72 400, `kcal` 400).
- `scripts/teams.py`: matches each side's six to its panel's team file and reads set familiarity against the pool
  (719 teams, `main.anchors.session.pool_team_texts`). All 10,800 sides matched uniquely.
- `scripts/analyze.py` → `results.json` (every number above). Run: `PYTHONPATH=<repo>/src:scripts python
  scripts/analyze.py games.jsonl.gz results.json` (≈ 1 min, 0.6 GB; short lane).
- The per-game rows (10 MB) are not committed: `~/gen3ai_archive/piloting_gap_forensics_2026-10-26/games.jsonl.gz`,
  sha256 in `games_sha256.txt`.
