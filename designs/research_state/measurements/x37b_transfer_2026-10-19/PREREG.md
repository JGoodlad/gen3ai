# X37b — the in-house TRANSFER read: PREREGISTRATION (2026-10-19)

**Written and committed BEFORE any game of this read.** Spec: `designs/research_state/EXPERIMENT_BACKLOG.md` row DV-2 /
X37b; [`design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §3 (H1, H2), §3.2
(D0e, its +2.0 pp rule), §5.4 (GEN-M) and F-DV-19. Nothing below changes after a game is played; a deviation is
reported as a FINDING in the read, never edited here. Frozen inputs: [`x37b_registration.json`](x37b_registration.json)
(digest in §6), written by `scripts/x37b.py register`.

Abbreviations: pp = percentage points; CI = 95 % interval; P\* = X26 @ 280,000,170 (the X35 plateau checkpoint);
25M = X26 @ 25,000,027; home72 = our 72 sample teams (all in the training pool); H = `heldout72_hpfix`, F = `forum72`
(both never trained on); U = H ∪ F, the two weighted equally; CRN = common random numbers (the same team pair on the
same battle seed in every arm compared).

## 0. The question and why this design

**Does X26's progress between 25M and P\* carry over to PILOTING teams it never trained on?** H1 (piloting
memorisation) predicts that the progress is pool-specific: P\*'s gain over 25M is larger on home72 than on unseen teams.

**The caution this design is built around** (TSR FINDING / F-DV-19, a hypothesis): rated in X26-vs-X26 games, X26 puts
H only ~5 pp further below home72 than Kakuna does, while against Kakuna it loses 11–28 pp more piloting H. So a
SAME-FAMILY, SYMMETRIC meter (both sides piloting the same unfamiliar set, GEN-M) may under-measure a piloting
deficit, and a GEN-M "TRANSFERS" alone could be a false kill. Two in-house reads are therefore played, both on CRN, and
the H1 kill needs BOTH (§5):

1. **T_tot (GEN-M, §5.4)** — P\* vs 25M, mirrored pairs where BOTH sides pilot teams of ONE set (home72, or H, or F).
   The total transfer shortfall (piloting and facing together). The precise read.
2. **T_pilot (D0e-pilot shape, §3.2)** — a FIXED opponent (P\*, the strongest in-house punisher) always pilots a
   FAMILIAR home72 team; the measured ref pilots the unseen team. This is the in-house analogue of X37a's REVERSE cell,
   the cell shape that revealed the piloting deficit against Kakuna (our side unfamiliar, the opponent familiar). The
   same games give **T_face** (the ref FACES the unseen team) for free.

No external opponent is used (no Kakuna, so no opponent-training bias).

## 1. Players, checkpoints, compute

- **Checkpoints** (read-only copies in `~/dev/gen3ai-reads/x37b_2026-10-19/model/`, every sha256 re-hashed before every
  unit; sources `models/rb_x26_s2601/checkpoints/`):
  - 25M `checkpoint_25000027_steps.zip` sha256 `26fda38d0ca6…5072`
  - 100M `checkpoint_100000037_steps.zip` `e1858ffe8a7d…ccb2` (trajectory only)
  - 175M `checkpoint_175000114_steps.zip` `20af225e57b1…3302` (trajectory only)
  - **P\*** `checkpoint_280000170_steps.zip` **`814823c514fb…91ed`**
- **Engine:** `main.h2h`'s in-process Rust eval core (`H2HEngine` with the team-strength meter's SCHEDULED builder,
  `main.team_strength.scheduled.ScheduledTeams`, B-D7's machinery): greedy vs greedy, CPU eager, 48 envs, 2 core
  threads, ONE torch thread (`OMP_NUM_THREADS=1`; `designs/ops/testing.md` "One torch thread for the CPU plays"), release, `ffi`, the env core's 250-turn stall threshold (a DRAW). The player sits in seat
  p1 in both games of a pair; a pair is two games on ONE battle seed with the teams handed over. Every played game's
  teams are checked against the plan before its unit is written (`team_strength.x26_lane.unit_rows`).
- **Played from a detached checkout of the commit that lands this preregistration** (named in the read). Rule 5b: ONE
  lane, `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh`, `CUDA_VISIBLE_DEVICES=""`; the T28 halt marker must be
  clear before every unit.
- **Panels:** home72, heldout72_hpfix, forum72 from `designs/research_state/heldout_panels.json` (B-D3, `8989c137`),
  count and content digest checked on load and against the registration.

## 2. The cells (the opponent is ALWAYS P\*; the player R is the ref)

| cell | player | teams of a pair (player's in game 2k, opponent's) | what it gives |
|---|---|---|---|
| `HH_R` | R | (a, b), a ≠ b, both home72 | r_R(HH): R vs P\* on the sample ground |
| `UU_R` | R | (a, b), a ≠ b, both from ONE unseen panel; half the pairs H–H, half F–F | r_R(UU_H), r_R(UU_F) |
| `X_R` | R | (u, h): u from H (half) or F (half), h from home72 | game A (2k): R pilots u vs P\* on h; game B (2k+1): R pilots h vs P\* on u |
| `X_P` | P\* | the SAME pairs and seeds as `X_R` | A_P, B_P: the P\* arm of the cross cell (CRN partner of every `X_R`) |

Pairs are uniform draws seeded by (registration seed 3737, block, round) (`round_pairs`); a block's battle-seed base is
shared by every player of that block and round (`round_seed`), so `HH_25M` and `HH_100M` of round r play the same team
pairs on the same seeds, and `X_R` / `X_P` likewise.

## 3. The queue, its size and its stop rule

- A **round** = 96 pairs (192 games) in each of the round's cells. **PRIMARY** rounds (R = 25M): `HH_25M`, `UU_25M`,
  `X_25M`, `X_P`, interleaved in that order round by round. Then **TRAJECTORY** rounds (descriptive): for R = 100M, then
  175M, `HH_R`, `UU_R`, `X_R` on the primary rounds' keys.
- Registered: **14 primary rounds** (10,752 games: 1,344 pairs in each primary cell) and **2 trajectory rounds per ref** (2,304 games). The engineering smoke ran at 0.28–0.40 games/s on 24-pair units beside a routine gate and the maturity queue, so the trajectory, and possibly the last primary rounds, may not be reached.
- **Stop rule (outcome-blind):** no unit starts after **2026-10-19T17:45** (local time, ≈ 6 lane-hours after the lane starts); the queue also stops when done. The read
  uses only WHOLE rounds counted from round 0 (a round counts iff all its units are recorded and every earlier round
  counts), so any stop leaves a balanced prefix. Nothing is looked at before the read except the lane's log lines
  (unit name, pair count, seconds); win rates are not printed by the lane.

## 4. The estimands (R = 25M for the primary; scores from the PLAYER's side)

- r_R(HH) = mean pair score of `HH_R`; g_home = 0.5 − r_R(HH) = P\*'s edge over R on the sample ground.
- r_R(UU_S) likewise on S ∈ {H, F}; g_S = 0.5 − r_R(UU_S); g_U = (g_H + g_F) / 2.
- **T_tot_U = g_home − g_U** (GEN-M's g_P − g_U).
- ΔA_S = mean over `X` keys of S of (A_P − A_R): P\*'s gain over R when piloting the unseen team against P\* on home72.
  ΔB_S = mean (B_P − B_R): the gain when FACING the unseen team (piloting home72).
- **T_pilot_U = g_home − ΔA_U** (D0e's T_S = g_home72 − g_S with g_S = ΔA). This uses P\*'s own HH score = 0.5 by
  the symmetry of P\* vs P\* on home72 (both teams uniform from one set); the seat term δ = (A_P + B_P − 1) / 2 is
  reported to check that assumption.
- **T_face_U = g_home − ΔB_U** (D0e-face).
- Per-panel versions (H, F) of each are reported beside the pooled U.

**Interval (governs):** a two-way TEAM bootstrap, B = 2,000, seed 3738: each replicate resamples every panel's teams with
replacement and weights each pair by the product of its two teams' multiplicities (one index set for every estimand);
percentile 2.5 / 97.5. Co-read (descriptive): the pair-level normal interval (teams fixed).

## 5. The decision rules (deterministic; BAR = +2.0 pp, the doc's)

**H1 verdict (primary, U pooled):**

| verdict | condition |
|---|---|
| **KILLED** | T_tot_U's upper bound < +2.0 pp **AND** T_pilot_U's lower bound ≤ 0 (the F-DV-19 veto: the familiar-opponent piloting read shows no detected shortfall) |
| **SUPPORTED** (a shortfall is detected) | not KILLED, and T_tot_U's lower bound > 0 **or** T_pilot_U's lower bound > 0 |
| **UNDECIDED** | otherwise |

- KILLED is checked first: a shortfall detected above 0 but surely below 2 pp kills H1 *as a major cause*, and is
  reported as "small, detected".
- **Rule 8:** a bound within 1e-9 of the boundary it is compared with makes the verdict UNDECIDED.
- **Near ties:** the verdict is recomputed without every pair (cross cells: every KEY, both arms) that holds a CPU
  near-tie decision (|Δ log-prob| < 2e-5, the executor's flag). If the two verdicts differ, H1 = UNDECIDED.
- Reported beside it, with the doc's own class rules, checked in the doc's order (TRANSFERS first): GEN-M's class on
  (T_tot, g_U) per U / H / F (§5.4: TRANSFERS / PARTIAL / POOL-ONLY / UNDECIDED), and D0e's class on (T_pilot, ΔA)
  per U / H / F (§3.2: TRANSFERS / PARTIAL / MEMORISATION-DOMINATED / UNDECIDED). T_face and its D0e-face class are
  descriptive (H2 is not this unit's verdict; X37a found FACE_c ≈ 0).

**Descriptive only (no verdict):** the trajectory refs' T_tot, T_pilot, T_face (the shortfall of P\*'s progress since
100M and since 175M) with the 25M ref's estimates on the same rounds; the in-house reverse-cell level A_P (P\* piloting
unseen vs P\* on home72) set beside the TSR strength gap, as the F-DV-19 sensitivity check; the seat term δ; per-panel
splits; logit-scale versions.

## 6. Frozen inputs

`x37b_registration.json`, digest **`91af4446bfe6f735eba2caeaa880f8121ca3a50de0dec478776b018301570cb2`**. A p72-only engineering smoke (one round of 24 pairs per cell, p72 vs p72,
seed 9999, not a cell or panel of this read) was played before this commit to validate the lane and measure the rate;
its outcomes were not read (the read pipeline was exercised on it printing row counts only). A first smoke at 3 torch
threads was abandoned after one unit: beside the other agent jobs on the three agent cores, a 24-pair unit ran > 20 min
(OpenMP oversubscription); one torch thread fixed it.

## 7. Hazards declared in advance

1. **Same-family opponent (F-DV-19).** Both reads are in-house; T_pilot's familiar-opponent shape is the mitigation,
   not a cure. A KILLED verdict says "no transfer shortfall IN-HOUSE"; it cannot speak for Kakuna.
2. **Ceiling / scale.** All estimands are on the probability scale; P\* vs 25M is far from 0.5, and a compressed scale
   on one set would bias T. Logit versions are reported.
3. **One seed, one lineage, pre-plateau P\*** (X35's plateau call). PROVISIONAL.
4. **T_pilot's precision** is roughly half T_tot's (two games per CRN key, one arm against a strong opponent): its role
   is the veto and the D0e class, not the headline.
5. **Mirror pairs are not two independent games** (TSR FINDING 8); the team bootstrap clusters on teams.
