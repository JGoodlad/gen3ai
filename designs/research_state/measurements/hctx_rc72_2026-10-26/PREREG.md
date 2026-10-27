# H-CTX: does P\* keep its home-team edge on NEW teams built from FAMILIAR sets? — PREREG (2026-10-26)

Written and committed BEFORE any counted game. Spec: [`read_spec.json`](read_spec.json) (`main.reads`, rule
`score_bands`). Hypothesis source: [`piloting_gap_forensics_2026-10-26/README.md`](../piloting_gap_forensics_2026-10-26/README.md)
(`c758fcb8`), "H-CTX".

Abbreviations: pp = percentage points; P\* = `rb_x26_s2601` @280,000,170 (X26's plateau checkpoint); Kakuna = Metamon's
Kakuna, checkpoint 34; H = `heldout72_hpfix`; F = `forum72`; U = H, F and P72 with equal weight (X51's definition);
a SET = one mon's species plus its sorted moves, Hidden Power typed (`gen3_team_overlap_v1`, the forensics' "mon
tier"); the POOL = the 719 teams P\* trained on (`main.anchors.session.pool_team_texts`).

## 1. The question

X51: P\* beats Kakuna **0.598** on home72 (its own pool's sample teams) and **0.487** on U (unfamiliar teams). The
forensics found the premium goes with the TEAM: a familiar set on an unfamiliar team plays like Kakuna's. **H-CTX:**
P\* memorised how to play its particular six-mon pool teams (roles, which mon to preserve, win conditions), not
portable set knowledge. The alternative, **SET-LEVEL**: what P\* learned is per set, and a team of familiar sets keeps
the premium whatever the six are.

**The test.** Play P\* against Kakuna on **rc72**: 72 NEW teams, every set of which is one the pool holds (verbatim),
recombined so that no team shares more than 2 sets with any single pool team. Set familiarity is home72's; team
context is new.

## 2. The panel, rc72 (built, frozen, registered before this PREREG)

`scripts/build_rc72.py` (its docstring is the full rule), seed 7272, deterministic (`--check`). Registered as a
DIAGNOSTIC panel in `designs/research_state/diagnostic_panels.json` (content digest `fceb2ea1…`, manifest
`rc72_manifest.json`): played by name like any panel, NOT held out and NOT part of the near-neighbour exclusion rule.

- **Sets:** every mon block is copied verbatim from a pool team (one pool occurrence of the drawn species, uniform
  over occurrences, so a set is drawn as often as the pool repeats it). Pool repeats of each chosen set: median 19
  (home72's sets: 21).
- **Species:** home72's species counts are the profile (41 species); later slots are weighted by Smogon teammate lift
  (clipped to [¼, 4]), so teams stay coherent. rc72 holds 35 species; its species distribution is 0.109 total
  variation from home72's.
- **Distance (each a refusal at draw time):** ≤ 2 shared sets with ONE team of the pool and of every committed
  training corpus (`w_forum_v1`, `d3_setmut_v1`, `n1_proc_v1`): 71 / 72 teams sit at exactly 2 with the pool. ≤ 2
  with ONE team of every held-out panel (`heldout72`, H, F, P72, the sealed `forum216b`) and X61's `n1p72` (REVIEW
  A's revealed-fit rule refuses 3). ≤ 3 within rc72.
- **Legal:** the pinned `TeamValidator('gen3ou')`, level 100 by Showdown's pack, Species Clause, every Hidden Power
  playing its named type at 70 once validated, every move and item playable by the engine, the label path.
  Refusals over the draw: 116 overlap, 2 held-out fit, 1 Hidden Power; 71 teams accepted on their first attempt.

## 3. The read

- **Cell:** X51's regime. Symmetric (both sides pilot rc72; our byte copy and Metamon's cache set `gen3ai_rc72`,
  content-checked before every unit by `main.x35.kakuna.check_panel_cell`), greedy against greedy, CPU, 20-game
  role-balanced units (10 each seat), unit seed = team seed = seed base, the 250-turn forfeit, Kakuna checkpoint 34.
  Played by `main.reads` (anchor lane) from a clean detached worktree at the spec's `pin`.
- **Units:** group `rc72`, unit seeds **0–59**: 60 units, **1,200 games**. Wilson 95 % half-width at 0.55: ±2.8 pp.
- **Guard (the engine rule):** group `guard`, P\* unchanged on home72, H, F and P72 at unit seed **15** (4 units, 80
  games), compared GAME FOR GAME (half, index, our team, result, turns) with X51's banked rows
  (`x51_plateau_vs_kakuna_2026-10-20/rows.jsonl`, sha256 `27a8d7ed…`). X51 played at `63f97a57`; this read plays at
  the pin. An identical replay shows the two engines play these cells identically, which is what licenses reading
  rc72 against X51's banked home72 and U. (The belief-prior-mix read's guards replayed X51 game for game at seeds 0
  and 10; seed 15 is new evidence.) The guard is listed first, so its units are among the first played.

## 4. Size, cost, stop

64 units, 1,280 games. Rates measured on this box: X51 11.8 s / game uncontended; this evening's N1-P units 230–1,160 s
per 20-game unit (12–58 s / game) with the heavy slots shared; the brief cites 27–93 s / game. Two heavy slots are
shared with X61P's anchor lane (K = 2) and S lane and a FIFO queue of other agents' jobs, so this lane (`--concurrency
2`) should complete about **2–4 units an hour: 16–32 h of wall time**. Registered stop: `--hours 48` from the lane's
start. `allow_partial_at_stop: true`: a lane ended by the stop records the counted units, and the rule's
`min_unit_frac` 0.9 (54 of 60 rc72 units) still decides VOID.

## 5. The decision rule (`score_bands`, params in the spec)

On the pooled rc72 score s = wins / games over counted units (ties count as not won, X51's convention), compared as
EXACT fractions (no rounding error can sit on a boundary, rule 8):

| s | verdict | reading |
|---|---|---|
| **≥ 0.57** | **H-CTX KILLED** | the premium survives new team contexts: P\*'s edge is set-level |
| **≤ 0.52** | **H-CTX SUPPORTED** | the premium does not survive: it is tied to the pool's particular teams |
| between | **UNDECIDED** | |

**VOID** (no verdict) if any guard unit is not counted, is not stamped `greedy`, or does not replay X51's games
exactly; if a counted rc72 unit is not `greedy`; or if fewer than 54 of the 60 rc72 units are counted.

Descriptive, read by no verdict: Wilson 95 % on s; the unit-level dispersion ratio φ and the interval widened by
√φ; Newcombe s − home72 (X51 banked, 239 / 400) and s − U (X51 banked equal-weight mean 0.487); the fraction of the
home72 − U premium rc72 keeps, (s − U) / (home72 − U).

## 6. Predictions (written before play)

- **H-CTX** predicts s ≈ U, about 0.49 (SUPPORTED). **SET-LEVEL** predicts s ≈ home72, about 0.60 (KILLED).
- **Mine: s ≈ 0.52 (0.49–0.55), SUPPORTED or a low UNDECIDED.** The forensics put ≈ 0.09 of P\*'s 0.11 KOs-per-mon
  home edge with team context and ≈ 0.03 with set novelty; rc72 takes the team context away and keeps the sets
  familiar, so about a quarter of the 11-pp premium should survive (≈ 0.515). Against it: rc72 teams are drawn by home72's
  species profile from the most repeated pool sets, so they may be STRONGER than U's teams, which a symmetric cell
  mostly cancels.

## 7. Confounds and limits (stated before play)

1. **A symmetric cell mixes piloting and facing**, as every X51 cell does. On rc72, as on home72, Kakuna's mons are
   familiar sets too, so rc72 against home72 changes team context on BOTH sides, and against U it changes set
   familiarity on both sides.
2. **Kakuna's own response to rc72 is in the score.** Metamon trained on its own team sets and human replays; if it
   pilots recombined teams worse (or better) than home72 / U, s moves for a reason that is not P\*'s (the open
   assumption A-K). No Kakuna-vs-Kakuna strength cell is played (the forensics suggested one; this brief did not).
3. **One checkpoint of one run** (P\*), one panel draw (seed 7272).
4. **rc72 is not a random draw of "new teams from familiar sets".** The species profile and teammate lift are
   choices; the ≤ 2 constraint excludes the most common set pairs that co-occur in pool teams. A different draw rule
   could land elsewhere.
5. The forensics' secondary H-CTX prediction (the seen-set per-mon edge ≈ 0) is NOT registered here. The captures
   (`--capture-dir`, every panel unit) allow it post hoc.

## Registration

Pin `b044d214b` (the commit that lands the panel, its cell, the rule and this PREREG; nothing on the play path
differs at the registration commit). Spec `read_spec.json`, sha256 `b9b52d6e18457f98dd5247c1e05020fe2c16111784ff0d977b081514ec03c101`.
Played from a clean detached worktree at the pin, `--concurrency 2`, `--hours 48`.

## Amendment 1 (2026-10-26, before ANY rc72 game): same-engine controls (the engine-identity rule, `20875e96`)

**Why.** The engine-identity rule (`designs/tools/reads.md` §4) landed while the first two guard units were in
flight. X51's rows were played on engine E0; this read plays on main's current engine. The parity fixes since
10-20 change rare mechanics, so a game-for-game guard against X51 does NOT license comparing rc72 with X51's banked
home72 / U. The coordinator's ruling: matched controls on THIS read's engine, interleaved unit by unit with rc72,
and the verdict read relative to them. Its sizes, interval method and the INCONCLUSIVE rule are mine, below.

**State at the amendment.** The registered lane (`~/gen3ai_archive/hctx_rc72_read`, spec `b9b52d6e…`, pin
`b044d214b`) was HELD at 23:27 PT. It had started only two GUARD units (home72 and H at seed 15); no rc72 unit was
ever started, so no rc72 game has been played or seen. That state is SUPERSEDED: never recorded, its two guard units
reported only as a descriptive replay check. §3's guard, §5's rule (`score_bands`) and §4's size are replaced by:

- **Arms, one read, one engine** (`read_spec_am1.json`, rule `premium_recovery`, a new state; pinned at the commit
  that lands the rule, which is after `20875e96`, so every unit stamps its engine):
  - **rc72:** unit seeds 0–59, **60 units, 1,200 games** (unchanged);
  - **home72 (familiar control):** seeds 0–39, **40 units, 800 games**;
  - **F = `forum72` (unfamiliar control):** seeds 0–39, **40 units, 800 games**.
  140 units, 2,800 games. Two-thirds of rc72's size per control: equal sizes would be 3,600 games. At 40 units each,
  the premium's 95 % half-width is ≈ ±4.8 pp; with X51's home72 − F (6.8 pp) as the truth, P(its lower bound
  clears 0) ≈ 0.80 (at 30 units, ≈ 0.66).
- **Interleaving.** Seven anchor groups, in this order: rc72 0–19, home72 0–19, F 0–19, rc72 20–39, home72 20–39, F
  20–39, rc72 40–59. The lane plays unit index i of every group before index i + 1, so each round of 7 units is 3
  rc72 + 2 home72 + 2 F and a stop at any point leaves the arms in proportion.
- **The rule** (exact fractions on the pooled scores; ties count as not won):
  - premium P = home72 − F; recovery R = rc72 − F;
  - **INCONCLUSIVE** if P's 95 % interval lower bound is not above 0;
  - otherwise **H-CTX KILLED** if R ≥ 0.75 P (rc72 recovers ≥ 75 % of the home premium), **H-CTX SUPPORTED** if
    R ≤ 0.25 P, else **UNDECIDED**.
- **Intervals:** a percentile bootstrap over 20-game units, each arm resampled with replacement independently,
  B = 10,000, `random.Random(7273)` (deterministic). Reported for P, R, rc72 − home72 and R / P.
- **VOID** if any counted unit is not stamped `greedy`, any arm has fewer than 75 % of its planned units counted, or
  the seven groups do not share ONE known engine. The engine is recorded in the decision.
- **Secondary, cross-engine (read by no verdict):** §5's absolute bands (≥ 0.57 / ≤ 0.52) on rc72, beside X51's
  banked home72 0.598 and U 0.487, labelled cross-engine. Also descriptive: how many home72 / F units at seeds 0–19
  replay X51's games exactly.
- **Stop:** `--hours 72`, `--concurrency 2`, `allow_partial_at_stop: true` (the 75 % floor still applies). The N1-P
  read keeps priority on the heavy slots. At 2–4 units an hour, 35–70 h.

**Predictions (unchanged in substance).** H-CTX: rc72 ≈ F, R / P ≈ 0 (SUPPORTED). SET-LEVEL: rc72 ≈ home72, R / P ≈ 1
(KILLED). Mine: R / P ≈ 0.25–0.4, so SUPPORTED or a low UNDECIDED. F's premium is smaller than U's (X51: 6.8 vs 11.1
pp), which is why the resolution gate exists.

**Amendment 1 registration.** Pin `aaf7ced86` (lands `premium_recovery` and this amendment; after `20875e96`, so
every unit stamps its engine). Spec `read_spec_am1.json`, sha256
`93b3fa31c50bc7e9a446452053a4d78c95e6e2d20964c9abf8c8ed533350b244`; state `~/gen3ai_archive/hctx_rc72_am1_read`.

## Amendment 2 (2026-10-27, before any rc72 game counted): the panel is redrawn as rc72b

**Why.** The first Amendment-1 rc72 unit (`pstar__rc72__00`) failed twice with 0 games: the live server rejected
the team ("Aerodactyl has Hidden Power Bug, but its IVs are for Hidden Power Water"). §2's Hidden Power rule was too
loose: it accepted a set whose text names a type and carries NO IV line (all 31), because Showdown's batch validator
REWRITES such IVs, while the live server REJECTS the team. 737 of the pool's 1,912 Hidden Power sets are of that
class; home72 and F hold none; rc72 held 71 (almost every team). The lane was HELD at 01:11 PT. No rc72 unit was
counted; the Amendment-1 state (`~/gen3ai_archive/hctx_rc72_am1_read`) is superseded and never recorded (its
counted home72 / F units are discarded with it).

**The fix.** `rc72b`: the same builder, seed and rules (§2) with X37a's STRICT Hidden Power rule: refused if a Hidden
Power does not play its named type at 70 AS WRITTEN, or the validator rewrites its IVs
(`build_rc72.py --panel rc72b`; content digest `4feda9ef…`, manifest `rc72b_manifest.json`). Every rc72b Hidden
Power set is of the class home72 and F play. Its profile: 34 species, 0.116 total variation from home72's; pool
repeats of each set median 18 (home72 21); max shared sets with ONE pool or corpus team 2 (all 72); with held-out
panels ≤ 2; within the panel ≤ 3. `rc72` stays registered as RETIRED (the superseded specs still parse).

**Consequence (stated before any game).** rc72b excludes the pool's no-IV-line Hidden Power sets, as home72 and F
do; a recombined panel cannot use the 38 % of the pool's Hidden Power sets that P\* saw written that way. If those
sets play differently in training than in the live server (an open question this read reports as a FINDING), rc72b is closer to home72 on that axis.

**Everything else is Amendment 1's** (arms, sizes, interleave, rule `premium_recovery`, bootstrap, VOID, secondary
line, stop), with the panel `rc72b`: spec `read_spec_am2.json`, a new state `~/gen3ai_archive/hctx_rc72_am2_read`,
pinned at the commit that lands rc72b.

**Amendment 2 registration.** Pin `905c29f5b` (lands rc72b). Spec `read_spec_am2.json`, sha256
`84901059df89a23793e75b52fd846eb0395c49ebb2696f50274453c1b3f90666`; state `~/gen3ai_archive/hctx_rc72_am2_read`.
