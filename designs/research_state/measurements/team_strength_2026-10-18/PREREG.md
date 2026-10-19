# TSR — the team-strength rating meter's first read: PREREGISTRATION (2026-10-18)

**Written and committed BEFORE any TSR game.** Nothing below changes after a game is played; a deviation is reported as
a FINDING in the read, never edited here. Tool: `python -m main.team_strength` (`src/main/team_strength/`). Frozen
inputs: [`tsr_registration.json`](tsr_registration.json), registration digest
**`336b8221c4b4704355c9cca51c37ad83f91ed5e5d1c4afbcc23f4dc5d5736e6a`**, written by `register` from
[`tsr_spec.json`](tsr_spec.json).

Abbreviations: pp = percentage points; CI = 95 % confidence interval; Elo = 400·log10 odds (+100 Elo against the
reference mean = a 64 % expected score; 35 Elo ≈ 5 pp at 50 %); BT = Bradley-Terry; P = home72, H = heldout72_hpfix
(X37a's names).

## 0. Why

X37a stage 1 measured K(P, H) = 0.573: with Kakuna piloting both sides, home72 teams beat heldout72_hpfix teams 57 %.
X37a's piloting gap PILOT_c (+0.123) is corrected only by that set-level MEAN. This read rates EACH TEAM with a fixed
pilot, so that (a) we can say whether H's teams are "competitive" against the other sets, (b) the piloting gap can be
re-read at MATCHED team strength, and (c) a widened training pool can be drawn by strength.

## 1. The sets and the reference panel (frozen in the registration)

| set | source | files | distinct teams (strict identity) |
|---|---|---:|---:|
| home72 | `n0_endofrun_2026-09-27/teams_home72` (our `data/teams/sample`) | 72 | 71 |
| away | Metamon `competitive/gen3ou`, byte copies in `teams_away/` | 20 | 19 |
| heldout72_hpfix | `heldout72_hpfix_kakuna_2026-10-18/teams_heldout72_hpfix` | 72 | 72 |
| forum72 | X37a stage 2 (`x37a_eval_matrix_2026-10-18/teams_forum72`) | 72 | 72 |
| p72 | X37a stage 2 (`x37a_eval_matrix_2026-10-18/teams_p72`) | 72 | 72 |

- **Team identity** = the six STRICT mon sets (species, moves with Hidden Power typed, item, ability, nature, EVs;
  `main.anchors.team_overlap`). 300 distinct teams; 6 are in both home72 and away (one rated team, a member of both).
- **Reference panel** (24 teams): 8 each from home72, away, forum72, drawn by `panel.draw_reference` with seed 3800,
  panel digest in the registration. Ratings are ANCHORED on it: the panel's mean is 0 Elo for each pilot.

## 2. The pilots and the games

**Pilot A = X26** (`rb_x26_s2601` `checkpoint_161000169_steps.zip`, sha256 `bc6050d1…2992`, the read-only copy in
`~/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/model/`, re-hashed before every unit) on BOTH sides, greedy vs
greedy, on the in-process Rust eval core through `main.h2h`'s engine (CPU, eager, 48 envs, 3 core threads, 3 torch
threads, release build, the env core's 250-turn stall threshold = a DRAW). Every one of the 300 teams plays **10
mirrored pairs** (20 games), each against a different reference team (never itself), seed 3801: in a pair the player
seat pilots the rated team in game 1 and the reference team in game 2 on ONE battle seed. 3,000 pairs = 6,000 games,
in 20 units of 150 pairs, the pair order shuffled so a stopped run leaves every set partly rated.
- **Played from a pinned detached checkout of the commit that lands this tool** (named in the read). Every
  play-relevant tree (`src/agents/{observation,model,battle,inference}`, `src/rust_env`, `src/rust_sim/src`) is
  byte-identical to `63f97a57`, the commit of every earlier X26 read; the transport differs (the in-process eval core,
  not the websocket front end).

**Pilot B = Kakuna** (`metamon:Kakuna` ckpt 34) on BOTH sides, greedy vs greedy, the anchors' B-D9 asymmetric pair
cell from the same pinned checkout (`anchors` is byte-identical to `cb0f1769`, X37a's `kcal` build): side A (reported)
= Kakuna on a 24-team SUBSET of one set (`--our-teamset away`), side B = Kakuna on the reference panel
(`--teamset home`), `--declare-asymmetric-teams`, 20 role-balanced games per unit, `--team-seed = --seed-base =` the
plan's keyed seed (3803). Subsets: 24 distinct teams per set by `panel.subset` (seed 3802; away has 19, all used).
**15 units per set** (300 games per set, 1,500 games), interleaved set by set. Metamon draws each side's team itself;
the front end's capture names both teams and the winner (`kakuna_lane.parse_unit`). Port 9563.
- **This arm is NOT mirrored** (the anchors refuse `--mirrored-pairs` for a pair cell): each game is its own cluster,
  random teams on both sides, seats balanced by role. A declared deviation from the brief's "mirrored".

**Counting.** A pilot-A unit counts iff every pair's two games played the planned teams in the planned seats
(`h2h.play.score_games` + the plan check). A pilot-B unit counts iff `main.anchors.results.unit_counts` passes against
the registered per-side content digests AND every capture matches its declared set on both sides AND the captures'
winners agree game for game with `games.jsonl` (a capture draw may book as a loss of the reported side, X37a
FINDING 2). A pilot-B game of a team against ITSELF (a reference team drawn on both sides) carries no information and
is dropped (counted, reported). A unit that fails is retried once (the queue's 2 tries), never replaced otherwise.

**Rule 5b and the clock.** One lane, `nice -n 19 taskset -c 5-7,13-15`, under `scripts/ops/mem_cap.sh` (pilot A: 16
GB; pilot B: 12 GB per unit). Pilot A first, then pilot B. Games start only after X37a stage 2's queue has finished.
**No unit starts after 2026-10-19 02:30 local**; a lane stopped there is read at its counted units, and the shortfall
is a FINDING. No result is looked at before both lanes end (only unit counts / integrity).

## 3. The fit (`main.team_strength.fit`)

- **BT on team identity**, per pilot: P(a beats b) = sigmoid(r_a − r_b); a draw scores 1/2; MAP under independent
  N(0, 2²) priors (logits); Newton to a step < 1e-10; shifted so the reference panel's mean is 0; reported in Elo.
- **Bootstrap**, B = 2,000, `default_rng(3804)` (pilot A) and `default_rng(3805)` (pilot B): within each rated team,
  its clusters (pilot A: mirrored pairs; pilot B: games) with replacement, refit. Per-team CI = percentile.
- **Per-set mean** (two-stage): each replicate's refit ratings averaged over the set's members drawn WITH REPLACEMENT
  (`default_rng(3814)`), percentile CI; the game-only CI beside it. Pilot A's sets are all their members; pilot B's
  are the subsets. A difference of two set means uses the same replicates.

## 4. Decision outputs

**D1 — is heldout72_hpfix "competitive"?** For each pilot, Δ = mean(H) − mean(forum72) in Elo with its two-stage CI:
- **MATERIALLY WEAKER** iff the upper bound < −35 Elo;
- **COMPETITIVE** iff the lower bound > −35 Elo;
- otherwise **UNDECIDED** (and UNDECIDED if a bound lies within 1e-9 of −35 or the upper within 1e-9 of 0).
- `detected_weaker` (upper bound < 0) is reported beside it.
- **The PRIMARY verdict is pilot A's**: X26 never trained on H or forum72, so the two sets are EQUALLY unfamiliar to
  the pilot. Pilot B's is the cross-check.
- forum72 is the comparator because it is human-built, competitive and unseen by both pilots' training as far as we
  know (§5.1 of the diversity doc: "near-neutral"). The same classes are reported for H − home72, H − away,
  H − p72, p72 − forum72, forum72 − home72 and away − home72. **Under pilot A every comparison against home72 or away
  carries X26's own familiarity with those sets** (it trained on them) and is DESCRIPTIVE; under pilot B they are
  read as strength, with Kakuna's own familiarity UNVERIFIED (A-K's analogue).

**D2 — do the pilots agree on which teams are strong?** Over the teams rated by both pilots (the subsets, the
reference panel excluded): Spearman's ρ of the two pilots' ratings, a bootstrap CI (teams resampled, each pilot's
replicate ratings, `default_rng(3824)`). **AGREE** iff ρ's lower bound > 0; **NOT SHOWN** otherwise (UNDECIDED within
1e-9 of 0). Reported beside it: both reliabilities (1 − mean bootstrap variance / variance of the point ratings) and
the attenuation-corrected ρ / √(rel_A · rel_B) (descriptive: with ~12 games per subset team in pilot B its
reliability is low, which caps the observable ρ).

**D3 — X37a's piloting gap at MATCHED strength (no new games).** Games where X26 PILOTS a team against Kakuna on
home72 (the same opponent distribution in both groups):
- FAMILIAR = the banked S(P, P) rows (`k2_x26_kakuna_both_teamsets_2026-10-17`, cell `X26_greedy_h72`, n = 400, our
  team = the row's `our_team`);
- UNFAMILIAR = X37a's REVERSE cell (`x37a_eval_matrix_2026-10-18/rows_stage1.jsonl`, `read_cell == reverse`, n = 400,
  our team = `our_team_stem`).
- Logistic P(win) = sigmoid(a + g·U + b·z), z = the PILOTED team's rating (logits); a tie is not a win (X37a's
  convention). **PILOT_m** = mean over the pooled games of sigmoid(a + b z) − sigmoid(a + g + b z). Bootstrap
  B = 2,000, `default_rng(3805)`: piloted teams resampled within each group, then their games, z from the rating's
  replicate of the same index. **DETECTED** iff its lower bound > 0; **GONE** iff its upper bound < +0.030;
  otherwise UNDECIDED (and within 1e-9 of a boundary). The RAW gap on the same games is reported beside it.
- **PRIMARY: z from PILOT B** (Kakuna's ratings; only the games whose piloted team is in Kakuna's home72 / H
  subsets). A pilot that trained on neither side's teams rates strength without X26's familiarity in it; Kakuna's own
  familiarity with P vs H is UNVERIFIED (A-K's analogue) and stated with the verdict.
- **SECONDARY (descriptive, circular by construction): z from PILOT A** over all 71 + 72 teams. X26's rating of a team
  CONTAINS X26's piloting of it, so matching on it removes the familiarity it is meant to measure; it answers "does the
  X26-rated strength account for the reverse cell", not "is there a familiarity gap".
- **Cross-check of the strength term:** pilot B's mean(home72) − mean(H) beside X37a's K(P, H) = 0.573 (= +51 Elo).

**D4 — descriptive, no verdict.**
- the per-team table (both pilots), every set's mean and spread;
- pilot A − pilot B per set (a set-level read of X26's familiarity relative to Kakuna's);
- forum72's ratings by its manifest's tier (A / B) and category; H by its frozen `ladder_weights_v1` stratum;
- the pool-widening recommendation (§6) is a reading of D1-D4, not a test.

## 5. Known hazards, stated before any game

1. **A rating is the strength of a team AS ITS PILOT PLAYS IT.** Pilot A trained on home72 and away (and 16 of the 24
   reference teams are from them), so its ratings of those sets include its familiarity; this is why D1's primary
   comparator is forum72 and D3's primary z is pilot B's.
2. **Same-pilot mirror pairs are not independent games.** A pair's two games share a seed and often end the same way
   for the same team; the bootstrap clusters by pair.
3. **The reference panel is the yardstick.** A different panel moves every rating by a constant and changes the
   resolution in different strength ranges; ratings are comparable only within one registration.
4. **Pilot B is unmirrored and thinly measured per team** (~12 games per subset team); its set means are its useful
   output, its per-team ratings are noisy.
5. **PROVISIONAL throughout:** X26 is one checkpoint (161M) of a live, unplateaued run, one seed.

## 6. What the read reports

The per-set strength table (both pilots), D1-D3's classes with their numbers, D4, every unit that did not count with
its reasons, the pool-widening recommendation (which teams a widened training pool should draw from, read from forum72's
strength by tier / category and H's / P72's against forum72), and every hazard as a FINDING. Updates
`design_diversity_teams_opponents.md` §5 and its decision record, `UNDERSTANDING.md`, and a ledger entry
(MEASUREMENT · PROVISIONAL).
