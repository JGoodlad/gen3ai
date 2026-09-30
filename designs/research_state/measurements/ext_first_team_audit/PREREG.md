# F-LH-13 re-measure — the population loop's MANIPULATION CHECKS over ALL of each specialist's teams

Pre-registered 2026-09-30 before any registered battle is played. Author: the F-LH-13 audit worker.
No game of this design had been played when this file was written; its sha256 is recorded in every
row's `prereg_sha` column.

## Why
F-LH-13 (the live Python eval measured a multi-team stable opponent on its FIRST pinned team only).
The population-loop PRIMARY reads (gap(RB) − gap(RC), round 1; gap(RB2) − gap(RC2), round 2) are
UNAFFECTED — every reader's target is an unpinned generalist. The two MANIPULATION CHECKS are
affected: M1 (round 1, `a6637f29`, +8.50 pp [+1.76, +15.13], ABSORBED) and M2 (round 2,
`aa8d56ea`, +8.67 pp [+3.06, +14.19], ABSORBED) were computed from the generalists' `ext_` series,
which had each 5-team offense specialist piloting only team 0 (`9eb3abdc52876a63`,
`team_sha f36747ae7e`). ABSORBED is the pre-registered condition that separates branch N+ from
branch M ("the loop did not engage — Δ tested nothing about generalization").

## Cells (all CPU, greedy BOTH sides — the `ext_` series' own regime)
Specialist = PILOT on one of its own 5 pinned teams (the recorded pin, index order);
generalist = OPPONENT drawing from the full team pool (`untaught_meter`'s `PairedPool`).
- Round 1: generalists B1 = `ai_v13_22_popr1_loop/snapshots/snapshot_000100000032.zip` (the loop arm),
  C1 = `ai_v13_23_popr1_ctrl/snapshots/snapshot_000100000032.zip` (the no-exploiter control);
  specialists A = `ai_v13_18_teach5_offense_hidose/final_model.zip` (arm A, the 1.78x-dose offense
  exploiter of G0), S13 = `ai_v13_13_exploit5_offense/final_model.zip` (the first offense exploiter of G0).
- Round 2: B2 = `ai_v13_27_popr2_loop/snapshots/snapshot_000110000016.zip`,
  C2 = `ai_v13_28_popr2_ctrl/snapshots/snapshot_000110000016.zip`; specialists A, S13 and
  RB = `ai_v13_24_popr1_read_loop/final_model.zip` (round 1's reader of B, added to B2's set).
- n = 60 battles per (generalist, specialist, team): round 1 600 per generalist (original M1 n = 400),
  round 2 900 per generalist (original M2 n = 600). Battle (team ti, index j) uses the SAME sim seed,
  pool draw and policy seeds for B and C (common random numbers), seed base 0, `untaught_meter`'s
  `sim_seed` / `pool_sequence` / `policy_seeds`.
- Tree: pin `6eb9c776` (the arms' own code), its OWN release `sim_bridge`, cwd = that tree.

## Statistics (decided now)
- PRIMARY: M_all = rate(B) − rate(C) pooled over all specialists × 5 teams (equal n per team, so equal
  weight), generalist wins / finished battles, `best_response_gap.newcombe_diff_ci` (the original
  rule's interval). **ABSORBED iff the lower bound > 0** — the registered rule, unchanged.
  Secondary: a paired cluster bootstrap over the (specialist, team) cells (20,000 draws, seed 20260930).
- Per-specialist and per-team descriptors, and RB's own B2 − C2 cell (original +17.0 [+7.3, +26.3]).
- REPRESENTATIVENESS: each generalist's rate on team 0 vs teams 1–4, Newcombe.
- REPRODUCTION CHECK (consistency, not exact — see deviations): the re-measured team-0 rate, pooled
  over specialists, vs the banked same-cycle `ext_` row (B1 88/200, C1 76/200 at 100,000,032;
  B2 146/300, C2 109/300 at 110,000,016). PASS iff the Newcombe CI of the difference contains 0.
- A cell with > 25 % timeouts is INCONCLUSIVE, never scored. No level is read before the registered n.

## Outcomes and what they mean (decided now)
- M_all ABSORBED in both rounds → the N+ branch assignments stand; the manipulation claims are
  re-stated as all-teams numbers.
- M_all NOT ABSORBED in a round → that round's branch becomes M ("the loop did not engage"); the
  stopping-rule COUNT is unchanged (N+ and M both count), but the reading "absorption did not
  generalize" is withdrawn for that round and replaced with "not engaged".

## Declared deviations from the original read
- D1 one snapshot per arm (cycle 3 of 4 in round 1, cycle 4 of 4 in round 2 — the only retained
  eval snapshots inside the pooled window) instead of the two pooled cycles.
- D2 the generalist draws from the unbiased pool (`PairedPool`), not the live eval's
  `Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)`.
- D3 players are `RLPlayer(stochastic=False)` on both sides (argmax over the mask) rather than
  `EvalRLPlayer` for the generalist; the same greedy regime.
- D4 fresh seeds, so the team-0 cell is a consistency check against the banked row, not a replay.
