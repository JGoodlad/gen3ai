# X37b — the in-house TRANSFER read: does X26's progress carry over to teams it never trained on? (2026-10-19)

**Preregistered** ([`PREREG.md`](PREREG.md), commit `38c64909`, before any game; registration digest `91af4446…`).
No external opponent: every game is X26 against X26 (no Kakuna, so no opponent-training bias). CPU only, one lane on
rule 5b's cores. Spec: `EXPERIMENT_BACKLOG.md` DV-2 / X37b;
[`design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §3 (H1), §3.2 (D0e), §5.4
(GEN-M), F-DV-19.

Abbreviations: pp = percentage points; CI = 95 % interval (the registered two-way TEAM bootstrap unless named
otherwise); P\* = X26 @ 280,000,170 (sha `814823c5…`, X35's plateau checkpoint); 25M = X26 @ 25,000,027; home72 = our
72 sample teams (all in the training pool); H = `heldout72_hpfix` (replay-reconstructed, unseen); F = `forum72`
(human-built, unseen); U = H and F weighted equally; CRN = common random numbers (the same team pair on the same battle
seed in every arm compared).

## Answer

- **H1 (piloting memorisation): UNDECIDED** by the registered rule. It is neither killed nor supported.
- **Most of X26's progress transfers.** P\*'s edge over 25M is **+18.8 pp [+15.5, +22.0]** on home72 and **+14.8 pp
  [+11.6, +17.9]** on the unseen panels, about 79 % of the in-pool gain. The unseen gain is clearly above 0.
- **The total transfer shortfall** (GEN-M: both sides pilot teams of one set) is **T_tot = +4.0 pp [−0.7, +8.4]**.
  It is not surely below the 2 pp bar, and not surely above 0 once the TEAMS are treated as a sample.
  - With the teams held fixed (the pair-level co-read) it is [+1.6, +6.4], detected. So on THESE 216 teams, P\* gained
    somewhat less on unseen teams.
  - Whether that generalises to unseen teams in general is not resolved: team-to-team variation is large.
- **The PILOTING-specific shortfall is ~0, and wide.** In the familiar-opponent shape (the ref pilots the unseen team;
  P\* pilots a home72 team), T_pilot = **+0.4 pp [−5.8, +6.5]**. Its point estimate does not support H1. Its interval
  cannot kill it.
- **The point shortfall sits on the FACING side:** T_face = +3.6 pp [−3.1, +10.4], not resolved. This is the mirror
  of the Kakuna read, where the drop was piloting (X37a: PILOT_c +12 pp, FACE_c ≈ 0). That is descriptive.
- **F-DV-19 (self-play may under-measure the piloting deficit) gains support** (descriptive):
  - P\* piloting H against P\* on home72 scores 0.418, about Kakuna's own strength gap K(P, H) = 0.573 (−7 to −8 pp).
    P\* piloting F scores 0.493.
  - So in-house, P\* shows almost no piloting deficit beyond team strength. Against Kakuna, X26 showed +12 pp
    (X37a), and +11 to +28 pp at matched strength (TSR).
  - Either the deficit is Kakuna-specific, or a same-family opponent does not punish it. These reads cannot tell
    which.
- **PROVISIONAL:** one lineage, one seed, P\* = the plateau checkpoint of a run that has since ended.

## The design

Every game has P\* as the OPPONENT; the measured player R sits in seat p1. Mirrored pairs: two games on one battle seed,
teams handed over. CRN: every player of a block and round plays the same team pairs on the same seeds.

| cell | player | a pair's teams | gives |
|---|---|---|---|
| `HH_25M` | 25M | two home72 teams | r(HH): 25M vs P\* on the sample ground |
| `UU_25M` | 25M | two teams of ONE unseen panel (half H–H, half F–F) | r(UU_H), r(UU_F) |
| `X_25M` | 25M | (u unseen, h home72) | game A: 25M pilots u vs P\* on h; game B: 25M pilots h vs P\* on u |
| `X_P` | P\* | the same pairs and seeds | A_P, B_P: P\*'s arm of the cross cell |

- **T_tot_U = g_home − g_U**, where g_S = 0.5 − r(SS) is P\*'s edge on set S. This is GEN-M (§5.4).
- **T_pilot_U = g_home − ΔA_U**, where ΔA = A_P − A_R is P\*'s gain over 25M when piloting the unseen team against a
  FAMILIAR-team P\*. This is D0e-pilot (§3.2) in the reverse-cell shape that revealed the Kakuna piloting deficit:
  the design-around for F-DV-19.
- **T_face_U = g_home − ΔB_U** (D0e-face, descriptive).
- **H1 rule:**
  - KILLED iff T_tot's upper bound < +2.0 pp AND T_pilot's lower bound ≤ 0;
  - SUPPORTED iff not killed and either lower bound > 0;
  - else UNDECIDED.
  - Rule 8 and a near-tie re-read must agree.

**Played:** all 14 registered primary rounds, 1,344 pairs per primary cell, **10,752 games**, on the in-process Rust eval
core (`main.h2h`'s engine with the team-strength meter's scheduled builder; greedy, CPU eager, 48 envs, 2 core
threads, 1 torch thread, release, `ffi`). They ran from the detached checkout `38c64909` between 11:01 and 16:44 local,
at 0.30–0.76 games/s beside the other agents' jobs. The trajectory (2,304 games) followed, and finished at 17:36,
before the registered 17:45 stop.

## Results (primary, R = 25M)

| quantity | H | F | **U (pooled)** |
|---|---|---|---|
| r(HH) (25M vs P\* on home72) | | | 0.312 [0.280, 0.345] |
| g_home (P\*'s edge on home72) | | | **+0.188 [+0.155, +0.220]** |
| g_S (P\*'s edge on the unseen set, GEN-M) | +0.136 [+0.094, +0.181] | +0.160 [+0.112, +0.207] | **+0.148 [+0.116, +0.179]** |
| **T_tot = g_home − g_S** | +0.052 [−0.003, +0.109] | +0.028 [−0.032, +0.085] | **+0.040 [−0.007, +0.084]** |
| ΔA (P\* − 25M piloting the unseen team) | +0.163 [+0.091, +0.238] | +0.204 [+0.117, +0.289] | +0.183 [+0.129, +0.241] |
| **T_pilot = g_home − ΔA** | +0.025 [−0.053, +0.104] | −0.016 [−0.107, +0.079] | **+0.004 [−0.058, +0.065]** |
| ΔB (P\* − 25M facing the unseen team) | | | +0.152 [+0.091, +0.210] |
| T_face = g_home − ΔB | +0.015 [−0.077, +0.107] | +0.057 [−0.032, +0.143] | +0.036 [−0.031, +0.104] |
| A_P (P\* pilots the unseen team vs P\* on home72) | 0.418 [0.358, 0.477] | 0.493 [0.424, 0.562] | 0.456 [0.410, 0.500] |
| seat term δ = (A_P + B_P − 1) / 2 | +0.006 [−0.006, +0.019] | −0.001 [−0.010, +0.009] | +0.002 [−0.005, +0.010] |

**Classes (the doc's rules):**
- GEN-M is **UNDECIDED** on U, H and F alike: T_tot's upper bound is above 2 pp and its lower bound is below 0.
- D0e-pilot is **UNDECIDED** on U, H and F alike.
- **H1: UNDECIDED.** The near-tie re-read drops the 13 games with a CPU near-tie decision, i.e. 2 HH pairs, 5 UU pairs and
  5 cross keys. It gives T_tot +0.038 [−0.008, +0.083] and T_pilot +0.005 [−0.058, +0.066], also UNDECIDED, so the two
  verdicts agree.

**Co-reads (descriptive):**
- Pair-level normal intervals, teams fixed: T_tot_U [+0.016, +0.064], T_pilot_U [−0.033, +0.042].
- Logit scale: T_tot +0.18 [−0.03, +0.38], T_pilot −0.02 [−0.32, +0.26], T_face +0.17 [−0.11, +0.46].
- The scale does not change the picture: the total shortfall is a point estimate on the facing side, and the piloting
  shortfall is ~0.

**The P\*-vs-P\* home72 assumption holds.** T_pilot uses P\*'s own home72 score = 0.5 by symmetry. The seat term δ is
+0.2 pp [−0.5, +1.0], so no seat advantage is visible.

## Trajectory (descriptive)

Both trajectory refs completed their 2 registered rounds before the stop: 192 pairs per cell, 2,304 games, on the
same keys as primary rounds 0–1. The whole registered queue (68 units, 13,056 games) finished at 17:36. Descriptive
only, with the team bootstrap:

| ref R | g_home (P\*'s edge on home72) | g_U (on unseen) | T_tot_U | T_pilot_U | T_face_U |
|---|---|---|---|---|---|
| 25M (rounds 0–1 only) | +0.212 | +0.194 | +0.018 | +0.100 | +0.048 |
| 100M | +0.035 [−0.041, +0.107] | +0.036 [−0.039, +0.114] | −0.001 [−0.109, +0.105] | −0.030 [−0.198, +0.142] | +0.033 [−0.166, +0.223] |
| 175M | +0.018 [−0.054, +0.097] | −0.005 [−0.085, +0.079] | +0.023 [−0.090, +0.139] | +0.023 [−0.135, +0.191] | +0.003 [−0.178, +0.185] |

- **Almost all of P\*'s in-house edge over 25M was gained by 100M.** From 100M to 280M, P\* gains +3.5 pp on home72
  and +3.6 pp on unseen teams, and neither is detected.
- Nothing in the 100M → 280M segment is pool-specific at this resolution: the T_tot points are −0.1 and +2.3 pp.
  The intervals are ±10 pp, which is too wide to say more.
- Rounds 0–1 alone give 25M's T_pilot +10 pp, against +0.4 pp over all 14 rounds. Two rounds are noise at this size.

## Reading

1. **The plateau we waited for is not a memorisation plateau, in-house.** About four fifths of what X26 learned
   between 25M and 280M shows up on teams it never saw: +14.8 pp on unseen panels, against +18.8 pp on the pool's own
   sample teams.
   - If H1 were the dominant story, g_U would be near 0 and T would equal g_home. It is not.
   - This answers the backlog's "is the plateau partly a memorisation plateau?" with: at most partly, and the part
     is ≤ ~8 pp of an ~19 pp gain (the T_tot upper bound).
2. **H1 cannot be killed at this size, and the piloting read gives it nothing.**
   - The total shortfall's point estimate (+4 pp) is above the 2 pp bar. Its team-bootstrap interval is ±4.5 pp,
     wider than the planned ±2.3 pp because team-to-team variation dominates.
   - The piloting-specific read, the one built around F-DV-19, centres on 0.
   - Killing H1 at the 2 pp bar would need roughly 4× the teams, not more games on the same teams. The team
     bootstrap is team-limited: 72 + 144 teams.
3. **The in-house meters do not see the deficit Kakuna sees.**
   - Against Kakuna, X26 pilots unseen teams 11–28 pp worse than familiar ones at matched strength (X37a, TSR).
   - In-house, P\* pilots H about as far below 0.5 as H's strength gap predicts (0.418 vs Kakuna's 0.573 ⇒ −7 pp of
     strength). It pilots F (≈ home72 strength) at 0.493.
   - So either the deficit is specific to how Kakuna exploits our play of unfamiliar teams, or a same-family
     opponent cannot punish it (F-DV-19). X37b cannot separate the two.
   - **Consequence for GEN-M as a standing meter:** a GEN-M "TRANSFERS" must not be read as "no held-out
     problem". It travels beside the Kakuna reverse cell, as §5.4 already says.
4. **What this means for D1 (widening, X38).**
   - The in-house transfer is large, so widening is not needed to make progress transfer in-house.
   - X37a's Kakuna-measured piloting deficit is what D1 targets, and this read neither confirms nor removes it.
   - X38's rank in X35 is unchanged: its demotion condition needs X37b to read TRANSFERS, and it read UNDECIDED.

## Integrity

- **Every unit counted.** All 56 primary units passed the lane's checks before being written:
  - `h2h.play.score_games`'s mirror invariants;
  - every pair's two games played the PLANNED teams in the planned seats (`team_strength.x26_lane.unit_rows`);
  - the read re-derives each unit's pairs from the registration and refuses any mismatch.
- **Before every unit:** every checkpoint was re-hashed against the registration, and the T28 halt marker was clear.
- **Draws and near ties:** 13 games carried a CPU near-tie decision (dropped in the registered re-read; nothing moved).
- **No unit was replayed or lost.** The read uses whole rounds from round 0.
- **Compute:** 0 GPU; `CUDA_VISIBLE_DEVICES=""`; one lane `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh`
  (peak < 1 GB); nothing under `models/` or `data/`; no server started.

## FINDINGS

1. **The registered lane script failed at startup and was fixed before any game.** home72 holds ONE TEAM TWICE:
   `63eda9d8d491d6a4.txt` and `dbbfac7bd5.txt` pack to the same team (TSR saw this as "72 files, 71 distinct
   teams"). The team-strength scheduled builder refuses one team under two identities.
   - The fix (`canonical()` in `scripts/x37b.py`, this commit) plays the duplicate as the sorted-first id, records
     both the plan id and the played id, and leaves the plan, the registration and every draw unchanged.
   - The lane ran from a byte copy of the fixed script beside the state (sha256 `736a8bf2…`), not from the pinned
     tree's copy.
   - The bootstrap still resamples the 72 registered FILE ids, so that team carries double weight in home72. Its
     effect is ~1/72 of the home72 cells.
   - The first launch's log is kept as `lane_attempt1.log` in the reads dir.
2. **Engineering smokes ran before the prereg commit** (declared in PREREG §6): p72 vs p72 only, outcomes never read.
   The first, at 3 torch threads, ran > 20 min for one 24-pair unit beside the other agents' jobs (OpenMP
   oversubscription), so the registration sets ONE torch thread.
3. **The team bootstrap is much wider than the planning arithmetic** (T_tot ±4.5 pp vs ±2.3 pp). Team heterogeneity
   dominates at 72 + 144 teams. Any future GEN-M rule at a 2 pp bar needs more TEAMS per panel, or a pre-registered
   statement that the panel's teams are the population (the pair-level interval). That choice decides between
   "detected on these teams" and "unresolved in general" here.
4. **T_pilot assumes P\*'s own home72 score is 0.5** (the symmetry of P\* vs P\* on one set). The measured seat term
   is ≈ 0 (+0.2 pp [−0.5, +1.0]), so the assumption holds within the data.
5. **F-DV-19 is supported, not settled:**
   - in-house, P\* shows ~no piloting deficit beyond strength on H;
   - Kakuna shows +11–28 pp at matched strength.
   - A same-family opponent is the confound that cannot be removed in-house. A pinned-peer K2 player (B-D8) or a
     second external opponent would be needed to separate "Kakuna-specific" from "in-house blind".
6. **The trajectory is thin by design**: 2 rounds per ref (192 pairs per cell), the registered size. Its ±10 pp
   intervals support only the statement "most of the in-house gain came before 100M". It does not change the
   verdict.
7. No other hazard hit: no GPU, no server, nothing under `models/` or `data/`, T28 clear throughout, every
   checkpoint's sha256 verified before every unit.

## Files

- `PREREG.md`, `x37b_registration.json`: the registration.
- `scripts/x37b.py`: the plan, the lane, the read, a `rows` export (added after the lane ran; the lane's byte copy,
  sha256 `736a8bf2…`, differs only by it) and a `selfcheck` (the plan's invariants, the estimator on synthetic
  rows with a known answer, the verdict rules including rule 8).
- `results.json`: the registered read (primary, the near-tie re-read, the trajectory).
- `rows.jsonl`: one row per recorded pair (unit, round, cell, plan teams, played teams, both games' results and
  near-tie counts).
- **Not committed:** the unit files and logs in `~/dev/gen3ai-reads/x37b_2026-10-19/`.
