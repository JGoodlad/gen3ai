# BOUNDARY-1 at 50M: the off-pool belief meter (REPORTED, no verdict)

**Status: REPORTED, DESCRIPTIVE, PROVISIONAL** (one checkpoint a run, three runs a cell). No registration, no class, no verdict, nothing about
100M, no strength claim. Ledger: 2026-11-18, "BOUNDARY-1 OFF-POOL BELIEF AT 50M". Queue item: P7's meter item (`designs/endstate/design_offpool_belief.md`
§5: "the meter is a REPORTED guard on BOUNDARY-1's finals"; `designs/tools/offpool_belief.md`).

**Question.** Does the breadth lever (half of each side's fresh team draws from `n1_proc_v1`) improve the network's belief about the
unseen species, the unshown moves and the unstated items of opponents on teams it never trained on, even though breadth was STOPPED-FOR-HARM on
the game meter at 50M (ledger 2026-11-18 READ; PREREG §8, §16)? And the same for width (`d_model` 256 against 128)?

## Command (the meter's documented form; nothing else typed)

```bash
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1       # a worktree at d9d7a7d61, no GPU, no lease
B=~/gen3ai_archive/offpool_belief_2026-11-09/bank                           # the baseline's four banks, reused as the docs prescribe
python -m main.offpool_belief score --bank $B/home72 --bank $B/heldout72_hpfix --bank $B/forum72 --bank $B/p72 \
    --ckpt <checkpoint_N_steps.zip> --label <run> --mix 0.25 --out scores/<run>.json          # 12 times: scripts/score_one.sh
python -m main.offpool_belief table --arm C=… --arm W=… --arm Br=… --arm WB=… --versus C                 # tables/cells_vs_C
python -m main.offpool_belief table --arm OFF=<C,W> --arm ON=<Br,WB> --versus OFF                         # tables/breadth_main
python -m main.offpool_belief table --arm OFF=<C,Br> --arm ON=<W,WB> --versus OFF                         # tables/width_main
python -m main.offpool_belief table --arm W=… --arm WB=… --versus W                                      # tables/breadth_at_256
python -m main.offpool_belief table --arm Br=… --arm WB=… --versus Br                                    # tables/width_at_breadth
python -m main.offpool_belief table <each run as its own arm>                                            # tables/per_run
python scripts/contrasts.py tables/per_run.json tables/seed_contrasts.json                               # supplement: seed-level interval
```

Each job ran niced (19), on the agent cores (`python -m utils.core_split agent-cpus`), under `scripts/ops/mem_cap.sh` (4 GB; 6 GB for the first),
three at a time while the C and W phase-2 runs held the GPU. The meter's code is `d9d7a7d61` (the worktree's HEAD = origin/main); model config 170,
observation semantics 168 on both sides (no `obs_gap` rows). Scores: `scores/` (sha256 in `scores.sha256`). Banks: the 2026-11-09 baseline's, unchanged
(`score` re-checked each rows file and re-resolved each panel against the live registry; the registry sha256 `ea3917c2f1e6…` equals the banks').

## Inputs

The FIRST checkpoint at or after 50,000,000 steps of each run, read from main's `models/` (never written, never copied). Every sha256 equals
`record/look_50M/verdict.json` `checkpoints` (the look used these same files), re-hashed before scoring.

| run | steps | sha256 |
|---|---|---|
| `rb_b1c_s5101` | 50,000,412 | `f706243d811a3767ff7cc107eb09ced8d71e2e646c8a0c8902fee7fd4fdbbf8f` |
| `rb_b1c_s5102` | 50,000,041 | `c46ddab83c0b6473ea2e0a040bc321717cd89177271fd47977a7f771bfcacb45` |
| `rb_b1c_s5110` | 50,000,220 | `cbcccec7e4822a034f36586e11b47d2154b26b8d2fc046e0845753efb9bc171f` |
| `rb_b1w_s5103` | 50,000,086 | `84227794ad05919476d90ec9d65068ac10fb57a9d09b0d4b9fe1425d66aefa84` |
| `rb_b1w_s5104` | 50,000,080 | `e4f5d19199f11b8064877891d1cba4b442da3f329a5c91e65a64c9480abd6356` |
| `rb_b1w_s5111` | 50,000,117 | `5d11bcb7a42285069d9141e077c220eea89214e911fa359f13f343a3fc56016c` |
| `rb_b1br_s5105` | 50,000,082 | `09774228297dde6c4038c0ad980d797d7364cebb6077e38a95de4746d052b513` |
| `rb_b1br_s5106` | 50,000,139 | `756bddc0ce398c980fb78960b3aa12e5206a750aebc26ac746c895bc04909fc5` |
| `rb_b1br_s5112` | 50,000,126 | `426e44334e00d3826097a99838dd36978f327c7a003cc4046f8b5b167510a340` |
| `rb_b1wb_s5107` | 50,000,127 | `af79a96b281b8a26f585d293997c71e46aec84e155965fa5c869b687e2787e34` |
| `rb_b1wb_s5108` | 50,000,005 | `4d8685165595fd1a0ed7c5e83de6fc934278f760fc119f79deab041119d66412` |
| `rb_b1wb_s5113` | 50,000,201 | `d459e408dcede04afb78fcd5bcc9bca9ead7e8eb74e98d1a09dbe853b0b253da` |

Cells: C = `rb_b1c_*` (width 128, pool only), W = `rb_b1w_*` (width 256), Br = `rb_b1br_*` (128, breadth), WB = `rb_b1wb_*` (256, breadth).

| bank (panel) | what it is | rows | rows sha256 | panel content digest |
|---|---|---|---|---|
| home72 | IN-POOL reference (72 training-pool sample teams) | 6,785 | `5207d8c100dff97a…` | `e7f90019ed48029a…` |
| heldout72_hpfix (H) | OFF-POOL, ladder-replay teams, Hidden Power repaired | 6,636 | `21545e87b1fb3198…` | `f7604f01a330b48f…` |
| forum72 (F) | OFF-POOL, 72 human-built Smogon-forum teams | 6,475 | `0caf6037fee34441…` | `efa72e52ce56bedb…` |
| p72 (P72) | OFF-POOL, procedural Smogon-usage teams (`ou_random_teams.js`, seed 3721) | 6,643 | `af668a3f65fbbfae…` | `e59d2f4843c01659…` |

The rows are the 2026-11-09 pilot's (P\*'s `pstar_plateau`, 280M steps) greedy self-play states at seat 2, the seat FACING the panel team; they are not
these runs' own states.

## What the numbers are

Every figure is **belief minus the Smogon prior, in nats a unit, on the meter's own units** (higher is better; the prior's level is in its own column;
it is identical for every run, same banks). A panel's number is the mean over its 72 teams of each team's mean. **The meter's interval is its
documented one: a team-clustered bootstrap, 10,000 draws, 95 %, one index matrix shared by every column (so a difference is paired on teams); an arm of
several checkpoints averages them inside each team first; the table also prints the per-checkpoint range.** That interval contains the spread over
*teams* and none of the spread over *runs*. The docs prescribe no other statistic, so the tables below are the meter's; the second interval in the
contrast table is a SUPPLEMENT of this record, not the meter's: a t interval on the 2 x 2 of the twelve per-run panel numbers, 95 %, **8 df** (the
variance pooled over the four cells, each with 3 runs, 4 x (3 - 1) = 8 df; main effects use all twelve runs, the width matched across breadth's two
sides and the reverse; `scripts/contrasts.py`). The two have the same centre (the same arithmetic) and contain different spread; neither replaces the
other. No multiplicity correction (3 verdict metrics x 4 panels x 6 contrasts were computed); a 95 % interval excluding zero here is a description,
not a decision.

Verdict columns per the docs: `species_logp` (the unseen species), `moves_ll` (the proper score for the unshown moves; `moves_logp` is kept in the full
tables for continuity), `items_logp`. The rank metrics (`species_hit`, `moves_prec`, `items_top1`) and the `--mix 0.25` floor columns are in
`tables/*.md`. Units scored a panel (home72 / H / F / P72): species 3,950 / 4,316 / 3,954 / 3,918; moves 28,057 / 25,984 / 26,443 / 26,966; items
7,887 / 7,198 / 8,013 / 9,437.

## Table: belief − prior, per cell (team-clustered 95 % interval, then the range of the three runs' own panel numbers)

**Read H and P72 for Br and WB as CONTAMINATED (PREREG §16, below): they are not generalisation. F is the panel to quote.**

#### Per cell, belief - prior, team-clustered 95% interval [seed range of the three runs]

| metric | panel | prior level | C | W | Br | WB |
|---|---|---|---|---|---|---|
| species | home72 (in pool) | -1.84 | +0.98 [+0.85, +1.11] [+0.92..+1.02] | +0.99 [+0.85, +1.12] [+0.98..+1.01] | +0.61 [+0.49, +0.74] [+0.59..+0.64] | +0.55 [+0.42, +0.67] [+0.54..+0.55] |
| species | H | -2.08 | -1.24 [-1.78, -0.74] [-1.30..-1.12] | -1.05 [-1.53, -0.61] [-1.14..-0.99] | -0.01 [-0.13, +0.10] [-0.03..+0.00] | -0.02 [-0.14, +0.09] [-0.04..-0.00] |
| species | F | -2.25 | -1.56 [-2.12, -1.06] [-1.65..-1.37] | -1.27 [-1.74, -0.86] [-1.36..-1.20] | -0.11 [-0.23, +0.00] [-0.19..-0.06] | -0.07 [-0.16, +0.02] [-0.14..-0.02] |
| species | P72 | -3.03 | -3.78 [-4.38, -3.20] [-4.00..-3.54] | -3.25 [-3.78, -2.73] [-3.40..-3.15] | +0.03 [-0.07, +0.14] [-0.00..+0.08] | +0.09 [-0.02, +0.20] [+0.06..+0.10] |
| moves_ll | home72 (in pool) | -4.80 | -0.35 [-0.61, -0.09] [-0.46..-0.16] | +0.23 [+0.04, +0.44] [+0.16..+0.29] | +0.15 [-0.01, +0.31] [+0.11..+0.18] | +0.29 [+0.17, +0.40] [+0.24..+0.36] |
| moves_ll | H | -4.97 | -1.69 [-1.87, -1.51] [-1.94..-1.27] | -0.77 [-0.89, -0.65] [-0.89..-0.69] | -0.59 [-0.70, -0.49] [-0.74..-0.48] | -0.13 [-0.22, -0.03] [-0.16..-0.08] |
| moves_ll | F | -5.11 | -0.92 [-1.10, -0.74] [-1.08..-0.78] | -0.21 [-0.34, -0.07] [-0.27..-0.12] | -0.28 [-0.40, -0.17] [-0.39..-0.23] | +0.08 [-0.01, +0.19] [+0.00..+0.17] |
| moves_ll | P72 | -5.60 | -1.87 [-2.04, -1.70] [-2.22..-1.60] | -0.91 [-1.01, -0.80] [-1.14..-0.75] | -0.71 [-0.84, -0.58] [-0.85..-0.60] | -0.17 [-0.27, -0.08] [-0.19..-0.14] |
| items | home72 (in pool) | -0.46 | +0.15 [+0.08, +0.23] [+0.13..+0.17] | +0.15 [+0.08, +0.22] [+0.14..+0.15] | +0.10 [+0.03, +0.17] [+0.09..+0.10] | +0.10 [+0.04, +0.17] [+0.10..+0.11] |
| items | H | -0.48 | -0.05 [-0.14, +0.02] [-0.07..-0.03] | -0.05 [-0.14, +0.03] [-0.08..-0.03] | +0.01 [-0.04, +0.07] [+0.01..+0.02] | +0.02 [-0.04, +0.08] [+0.01..+0.02] |
| items | F | -0.74 | +0.08 [-0.05, +0.21] [+0.06..+0.10] | +0.07 [-0.05, +0.20] [+0.06..+0.10] | +0.11 [+0.03, +0.21] [+0.09..+0.15] | +0.11 [+0.02, +0.20] [+0.08..+0.12] |
| items | P72 | -0.63 | -0.20 [-0.38, -0.06] [-0.24..-0.17] | -0.19 [-0.37, -0.04] [-0.21..-0.18] | +0.10 [+0.05, +0.15] [+0.09..+0.11] | +0.10 [+0.05, +0.16] [+0.10..+0.11] |

#### Factor contrasts, meter team-bootstrap 95% interval | seed-level 95% t interval (8 df)

| metric | panel | breadth ON-OFF (Br+WB vs C+W) | seed t | width ON-OFF (W+WB vs C+Br) | seed t | breadth @128 Br-C | seed t | breadth @256 WB-W | seed t |
|---|---|---|---|---|---|---|---|---|---|
| species | home72 (in pool) | -0.40 [-0.46, -0.35] | -0.40 [-0.44, -0.36] | -0.03 [-0.05, -0.00] | -0.03 [-0.07, +0.01] | -0.37 [-0.43, -0.30] | -0.37 [-0.43, -0.31] | -0.44 [-0.51, -0.37] | -0.44 [-0.50, -0.38] |
| species | H | +1.12 [+0.72, +1.56] | +1.12 [+1.04, +1.21] | +0.09 [+0.02, +0.17] | +0.09 [+0.01, +0.17] | +1.22 [+0.79, +1.69] | +1.22 [+1.10, +1.34] | +1.03 [+0.64, +1.44] | +1.03 [+0.91, +1.14] |
| species | F | +1.32 [+0.93, +1.78] | +1.32 [+1.18, +1.46] | +0.16 [+0.06, +0.28] | +0.16 [+0.03, +0.30] | +1.45 [+1.02, +1.94] | +1.45 [+1.25, +1.64] | +1.20 [+0.83, +1.62] | +1.20 [+1.00, +1.39] |
| species | P72 | +3.57 [+3.02, +4.16] | +3.57 [+3.39, +3.76] | +0.30 [+0.20, +0.40] | +0.30 [+0.11, +0.48] | +3.81 [+3.21, +4.45] | +3.81 [+3.56, +4.07] | +3.33 [+2.81, +3.89] | +3.33 [+3.07, +3.59] |
| moves_ll | home72 (in pool) | +0.27 [+0.17, +0.38] | +0.27 [+0.15, +0.40] | +0.36 [+0.26, +0.47] | +0.36 [+0.23, +0.48] | +0.50 [+0.36, +0.65] | +0.50 [+0.32, +0.68] | +0.05 [-0.07, +0.17] | +0.05 [-0.13, +0.23] |
| moves_ll | H | +0.87 [+0.75, +0.99] | +0.87 [+0.60, +1.14] | +0.69 [+0.61, +0.78] | +0.69 [+0.42, +0.96] | +1.09 [+0.93, +1.26] | +1.09 [+0.71, +1.47] | +0.64 [+0.53, +0.76] | +0.64 [+0.26, +1.02] |
| moves_ll | F | +0.46 [+0.37, +0.56] | +0.46 [+0.33, +0.60] | +0.54 [+0.46, +0.62] | +0.54 [+0.41, +0.68] | +0.64 [+0.51, +0.78] | +0.64 [+0.44, +0.83] | +0.29 [+0.20, +0.38] | +0.29 [+0.10, +0.48] |
| moves_ll | P72 | +0.95 [+0.83, +1.07] | +0.95 [+0.68, +1.21] | +0.75 [+0.66, +0.84] | +0.75 [+0.48, +1.02] | +1.16 [+1.00, +1.33] | +1.16 [+0.78, +1.54] | +0.73 [+0.61, +0.86] | +0.73 [+0.36, +1.11] |
| items | home72 (in pool) | -0.05 [-0.10, -0.01] | -0.05 [-0.07, -0.04] | -0.00 [-0.01, +0.01] | -0.00 [-0.02, +0.01] | -0.06 [-0.11, -0.01] | -0.06 [-0.08, -0.04] | -0.05 [-0.09, -0.00] | -0.05 [-0.07, -0.02] |
| items | H | +0.07 [+0.00, +0.14] | +0.07 [+0.04, +0.09] | +0.00 [-0.01, +0.02] | +0.00 [-0.02, +0.03] | +0.07 [+0.01, +0.13] | +0.07 [+0.04, +0.10] | +0.07 [-0.00, +0.15] | +0.07 [+0.03, +0.10] |
| items | F | +0.03 [-0.04, +0.12] | +0.03 [-0.00, +0.07] | -0.01 [-0.03, +0.01] | -0.01 [-0.04, +0.03] | +0.03 [-0.05, +0.13] | +0.03 [-0.01, +0.08] | +0.03 [-0.04, +0.12] | +0.03 [-0.02, +0.08] |
| items | P72 | +0.30 [+0.16, +0.47] | +0.30 [+0.27, +0.33] | +0.01 [-0.02, +0.03] | +0.01 [-0.02, +0.04] | +0.30 [+0.16, +0.48] | +0.30 [+0.26, +0.34] | +0.30 [+0.15, +0.47] | +0.30 [+0.26, +0.34] |


Every breadth contrast on the H and P72 rows is contaminated (§16 below); the width contrasts on every row, and every contrast on the home72 and F rows
(F with its own qualifier), are the ones to quote. A contrast's first interval is the meter's, the "seed t" is this record's supplement.

(Column order: the first interval is the meter's team-bootstrap for the cell mean of three runs, the second bracket is the minimum..maximum of the three
runs.)

## Reading it (descriptive)

- **Species belief (F, the clean panel):** C -1.56 [-2.12, -1.06] and W -1.27 [-1.74, -0.86] are WORSE than the prior; Br -0.11 [-0.23, +0.00] and WB
  -0.07 [-0.16, +0.02] are at the prior (NOT DETECTED either way). Breadth ON minus OFF is **+1.32 [+0.93, +1.78]** (team) / [+1.18, +1.46] (seed t,
  8 df); width ON minus OFF is +0.16 [+0.06, +0.28] / [+0.03, +0.30]. **No arm is detected ABOVE the prior off the pool on the species leg**; breadth removes
  the deficit, it does not add to it (the same result as the baseline's N1-P, the docs' "breadth brings it to the prior"). In the pool, breadth costs
  **−0.40 [−0.46, −0.35]** nats of species advantage on home72 (C +0.98, Br +0.61).
- **Move belief (`moves_ll`, F):** every cell is WORSE than the prior except WB (+0.08 [−0.01, +0.19], NOT DETECTED); C −0.92, W −0.21, Br −0.28.
  Breadth ON−OFF **+0.46 [+0.37, +0.56]** / [+0.33, +0.60], and width ON−OFF is as large, **+0.54 [+0.46, +0.62]** / [+0.41, +0.68]: the move leg
  responds to width at least as much as to breadth, and the two sum (WB − C = +1.0 on F). This is the leg the baseline found breadth did NOT repair
  (on `moves_logp`, a column blind to over-confidence; the baseline predates `moves_ll`, so no baseline number exists for it).
- **Item belief (F):** breadth ON−OFF +0.03 [−0.04, +0.12] / [−0.00, +0.07]; width ON−OFF −0.01: NOT DETECTED either way on F. Items differ on P72
  (+0.30, §16 contaminated) and H (+0.07 [+0.00, +0.14], contaminated).
- **The seed spread is small next to the team interval** on every verdict metric except `moves_ll` (e.g. species F, C: runs −1.65 .. −1.37; `moves_ll`
  H, C: −1.94 .. −1.27, the widest), so the 3-vs-3 seed-level interval is narrower than the team interval for species and items and WIDER than it
  for `moves_ll` (H breadth ON−OFF: team [+0.75, +0.99], seed [+0.60, +1.14]).
- **Two meters, two signs, no link claimed.** Breadth was STOPPED-FOR-HARM on the game meter (HF −5.18 pp); on the belief meter it reads ahead off the
  pool on the species leg (to the prior) and on `moves_ll`, not detected on items at F, and behind in the pool on species and items. The docs' standing sentence applies: no game read has DETECTED a
  win-rate effect of belief quality (ledger 2026-10-26), so this record neither explains the stop nor argues against it.

## PREREG §16: the contamination of breadth's off-pool read, exactly

§16: "Breadth's read on H, P72, U and N is contaminated by its own training teams and is never quoted as generalisation." What the meter reads and
how it overlaps breadth's pool `n1_proc_v1` (`tier ALL`, 12,000 teams, data sha256 `99e3425a…0bad6`; `panel_overlap_n1_proc_v1.json`,
`scripts/panel_overlap.py`, using `main.anchors.team_overlap`, the repository's own overlap counter):

- **Panels the meter reads (all four registered):** home72, H, F, P72 (the table above). The scored units are the hidden species, the unshown moves and the
  unstated items of the opponent on the PANEL team, i.e. mons and sets of the panel teams.
- **Overlap with `n1_proc_v1`, by FULL mon set (species, moves, item, ability, nature and EVs all equal; PREREG §5's "62 %" and "56.5 %" are this
  count, reproduced):** H **269 / 432 = 62.3 %**, P72 **244 / 432 = 56.5 %**, home72 183 / 432 = 42.4 % (training-pool teams), **F 84 / 432 = 19.4 %**.
  By species + moves only (the counter's `mon` tier): H 410, F 359, P72 370, home72 397 of 432 (83 to 95 %). **No panel team is an exact copy of a corpus
  team, and none shares four or more mon sets with ONE corpus team (the registry's near-neighbour rule held on all four)**; the six species of a team
  equal a corpus team's six species for 16 of 72 (H), **18 of 72 (F)**, 6 of 72 (P72), 20 of 72 (home72).
- So: **H and P72 are contaminated for the breadth arms (Br, WB)** at set level, P72 further because `p72` is built by the same generator family as the
  corpus (`designs/tools/offpool_belief.md`); their breadth cells and every breadth-ON − OFF contrast on them are listed in the tables and are NOT
  quoted as generalisation. **F is the least overlapped panel, not an uncontaminated one:** one mon set in five, and 18 of its 72 teams share all six
  species with some corpus team, are in the pool the breadth arms drew from (and the corpus is drawn from the same Smogon tables the prior is built
  from, which is why a breadth-trained belief is expected to land on the prior: `designs/tools/offpool_belief.md`, "What it is NOT"). The F numbers
  carry that qualifier. The C and W arms trained on the 719-team pool only; the registry's guard is what makes F, H and P72 off-pool for them, and the
  meter does not re-derive what each run's `--team-mix` drew.
- Width's contrast (W + WB against C + Br) has one breadth-trained and one breadth-free arm on each side, so the corpus overlap cancels in it (PREREG
  §5, hazard M4), on every panel.

## Comparison with the 2026-11-09 baseline (only as the docs allow)

Same banks, so the **prior's level** reproduces the baseline's (home72 −1.84, H −2.08, F −2.25, P72 −3.03 on species). The baseline arms (P\*, C0, N1-P,
X60-T) are continuations of a 280M-step run at 20M steps; BOUNDARY-1's are from-scratch at 50M with another observation/config version (170 against
158/161), so they are NOT comparators and no arm-minus-arm number is formed across the two (also: root `CLAUDE.md`, "pre-era runs are history, never
comparators"). For orientation only: the baseline's pool-only C0 read −4.02 on F species and its breadth arm N1-P −0.07; these runs read −1.56 (C) and
−0.11 (Br) on F. The baseline has no `moves_ll` for any arm (its scores predate the column).

## Cost

Per checkpoint (four banks, ≈ 26,500 rows, three jobs at once on the agent cores while a run held the GPU), from `logs/time_*.txt` (`/usr/bin/time -v`):
width 128: 212 to 238 CPU-s user (3.5 to 4.0 CPU-min), 3:33 to 3:59 wall; width 256: 332 to 345 CPU-s user (5.5 to 5.8 CPU-min), 5:35 to 5:51 wall.
**Peak resident 1.17 GB (width 128), 1.23 GB (width 256)**; mean per-checkpoint cost 4.8 CPU-min against the docs' "≈ 8 CPU-min". The twelve took
about 23 minutes of wall in all (the first alone, then three at a time). A 2 GB cap would fit (UNVERIFIED: 4 GB was used, 6 GB for the first).

## Caveats and FINDINGs

1. **One checkpoint a run, 50M, three runs a cell.** Nothing about 100M, nothing about the plateau, nothing about C or W's phase-2 states.
2. **The rows are the plateau pilot's greedy states, not each run's own** (the meter's standing caveat); a belief can read better on those rows and
   worse on the states the run itself reaches.
3. **Not a strength number.** No game read has DETECTED an effect of belief quality on win rate; breadth reads better here and worse in games.
4. **Breadth's F read is not contamination-free** (19.4 % of F's full mon sets and 18 of 72 teams' six species are in `n1_proc_v1`; above) and H and P72 are
   contaminated; the prior-landing of a breadth arm on F is expected from a corpus drawn from the same Smogon tables as the prior.
5. **Two intervals, neither complete.** The meter's contains the team spread and no seed spread; the supplement contains the seed spread (3 runs a
   cell, 8 pooled df, the variance estimate itself from 12 runs) and no team spread. The supplement's cell SDs are from three numbers.
6. **No multiplicity control** (see above) and no pre-registered threshold; nothing here is "better", "worse" or "detected" in the registered sense;
   the meter's BETTER / WORSE words in `tables/*.md` are its own interval-vs-zero labels.
7. **An earlier, unrecorded scoring exists** at `~/gen3ai_archive/b1_belief_meter_2026-11-17/` (nine runs, one checkpoint each, `final_model.zip`
   dated 2026-11-13; commit 81594609; two seeds each for W, Br and WB, three for C). It is in no ledger entry, doc or record that this session could find; it was not used or compared here. (UNVERIFIED which step each of its
   files is at; not read beyond its logs and one score header.)
8. **Bank provenance** (the baseline's FINDING 6, unchanged): the banks were built from the tool's tree before it was committed (stamped `3b1733f50`,
   dirty); they re-check against the live registry and the rows' sha256 on every `score`.
9. `per_run.json` / `seed_contrasts.json` are included for reproducibility of the supplement; `tables/*.json` are the meter's own documents.

## Files

`scores/` (12 score files + `scores.sha256`), `tables/` (the meter's `table` output, markdown and JSON, for each command above; `digest.md` is the compact
view used here), `tables/seed_contrasts.json` (the supplement), `panel_overlap_n1_proc_v1.json`, `scripts/` (the jobs, `contrasts.py`, `digest.py`,
`panel_overlap.py`, `ckpts.txt`), `logs/` (`driver.log`, one `/usr/bin/time -v` file a job). Full working copy: `~/gen3ai_archive/b1_offpool_belief_50M_20261118/`.
