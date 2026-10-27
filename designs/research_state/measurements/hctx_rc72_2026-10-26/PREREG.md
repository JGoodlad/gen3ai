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
