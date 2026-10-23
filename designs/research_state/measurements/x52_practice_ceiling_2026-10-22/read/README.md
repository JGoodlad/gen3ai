# X52's registered read: the practice ceiling against Kakuna (2026-10-22/23)

**Question** (registered in the X35 launch sheet's §0, AM-X52; design: `design_beat_metamon_path.md` §6.2). Does
PRACTISING a team fix our piloting of it against Kakuna? Does practising 36 teams transfer to 36 unpractised teams of
the same population?

Abbreviations: pp = percentage points; Δ = (s3501 @5M + s3502 @5M, pooled) − P\*; LB / UB = the 95 % interval's
bounds; n.d. = not detected. P\* = `rb_x26_s2601` @280,000,170.

## Answer: **UNDECIDED** (no rule held)

| half | 5M seeds pooled | P\* | **Δ, team-clustered 95 %** | registered prediction (80 %) |
|---|---|---|---|---|
| **practised** | 0.553 (n = 1,200) | 0.522 [0.482, 0.561] (n = 600) | **+3.2 pp [−0.2, +6.8]** | +5 [+0.5, +9.5]: inside |
| **unpractised** | 0.572 (n = 1,200) | 0.558 [0.518, 0.598] (n = 600) | **+1.3 pp [−3.5, +6.1]** | 0 [−3.5, +3]: inside |

- **The class, by AM-X52-g in order.** INCONCLUSIVE: no (0 % uncounted). NO FAMILIARITY: no (UB_pr +6.8 ≥ +4.0).
  TRANSFERS, NO TRANSFER, FAMILIARITY ONLY: no (LB_pr −0.24 pp ≤ 0). So **UNDECIDED**. No bound sits within 0.0005 of
  its boundary (rule 8).
- **Against the registered class odds:** UNDECIDED was the modal class (0.40). Both point estimates are inside their
  80 % bands. The practised gain is about 2 pp under the central guess and is not resolved from zero.
- **Consequence (AM-X52-h):** D1 proceeds as staged.
- **The intervals came in narrower than registered:** half-widths ≈ 3.5 pp (practised) and 4.8 pp (unpractised),
  against the planned ±4.9.

**Per seed** (descriptive, the same clustered bootstrap):

| | practised | unpractised |
|---|---|---|
| s3501 @5M − P\* | +3.7 [−0.4, +7.9] | +2.3 [−2.7, +7.2] |
| s3502 @5M − P\* | +2.7 [−1.4, +6.9] | +0.3 [−5.8, +6.6] |

## The descriptive cells (P-X52-4)

- **s3501 @1M on the practised half:** 0.555 (n = 200). On the same 10 units, 1M − P\* = **−3.5 pp [−12.1, +4.6]**
  and 5M − P\* = +3.0 [−4.6, +10.3]. Predicted +1 at 1M. The dose curve is too noisy at 200 games to read.
- **C0 s3501's final (P\* + 20M of plain self-play) on the unpractised half:** 0.528 (n = 400). On the same 20 units,
  C0 − P\* = **−2.3 pp [−7.9, +3.6]**. Predicted 0 ± 3: inside. Plain continuation shows no drift up on unseen teams.
  - So the unpractised half's +1.3 is not a continuation effect.
  - Not registered: X52 (both seeds) − C0 on the same 20 units is +4.0 pp (0.568 vs 0.528; no interval
    computed). That is a hint, no more, that practising other teams did not narrow our piloting of unseen
    ones (F-X52-1's worry).
- **P\* scores ≈ 0.50 on each half (predicted):** it read 0.522 and **0.558** instead. The unpractised half's LB is above
  0.50.
  - These are REVERSE cells: Kakuna pilots home72, not the unfamiliar half. So they are not comparable with X51's
    symmetric U = 0.487, where Kakuna pilots the same unfamiliar set we do.
  - Read with X51's home72 0.598, they again say the deficit against Kakuna is in what WE pilot, and that it shrinks
    when Kakuna pilots home72.

## Method (as registered: AM-X52-e, f, g)

- **The cells.** REVERSE cells: our model pilots one half (its 36 files), and Kakuna ckpt 34 (`Kakuna@ckpt34`,
  Metamon `0a00a759`) pilots home72 (`gen3ai_home72`).
  - Greedy against greedy, `main.anchors`, 20-game role-balanced units.
  - `--team-seed` = `--seed-base` = the unit seed, 0–29 for every primary (model, half) cell. So every model on a half
    plays the same deals: twins (unit seed, half, game index) carry the same our_team, 1,200 / 1,200.
  - 6 primary cells × 600 games, plus 200 (1M) and 400 (C0), make **4,200 games**.
- **The harness.** ONE fresh detached checkout of the X35 production pin `f4b25238`, the pin both X52 seeds and C0
  trained at.
  - It loads P\* (config v158) as the forks did, so every difference is a model difference only.
  - Server `ws_frontend[in-process]@f4b25238d+rust:cargo:src/rust_sim`, Showdown pin `e0551883f`, the 250-turn
    forfeit, CPU, one torch thread per side, `bare` load.
  - Each cell carries ONE regime stamp over all its rows.
- **The models.** Read-only copies, sha256 re-checked before every unit (`scripts/x52q.py` `MODELS`):

| model | file | step | sha256 |
|---|---|---|---|
| P\* | `checkpoint_280000170_steps.zip` | 280,000,170 | `814823c5…91ed` |
| s3501 @5M | `rb_x35_x52prac_s3501/final_model.zip` | 285,022,320 | `58f92e7f…c9d2a` |
| s3502 @5M | `rb_x35_x52prac_s3502/final_model.zip` | 285,021,316 | `c0058823…0ece0` |
| s3501 @1M | `checkpoint_281000024_steps.zip` | 281,000,024 | `07bc9af0…09fe` |
| C0 s3501 final | `rb_x35_c0_s3501/final_model.zip` | 300,061,940 | `09a6a2df…3303` |

  "@5M" is each run's `final_model.zip` (the arm end, X35's "final" convention). It is one PPO update past the 5M
  checkpoint file (F-X52-R2).
- **Teams.** The halves are byte copies checked against `x52_teams_manifest.json`, and Kakuna's home72 against
  `home72_manifest.json`'s `metamon_teams`, both before every unit.
- **Counting.** `main.anchors.results.unit_counts`, DECLARED mode (36-vs-72 by design): the summary's own team digests
  equal `session.team_set_digest` of the checked files.
  - Plus X51's CAPTURE check: each game's packed START teams are the row's half team and exactly one home72 team.
  - **210 / 210 units counted on their first try**: no partials, no fails.
- **Statistic.** Ties are non-wins. The interval is a percentile bootstrap that resamples the half's 36 teams with
  replacement: 10,000 resamples, `random.Random(20261022)`.
- **The F-X52-6 smoke** (2 games each: P\* on the practised half, s3501 @5M on the unpractised half) PASSED the
  counting rule and the capture check before the read. The reverse config is validated.

## Cost

- **Play:** 8.4 lane-h of play, 7.2 s / game. One lane on rule 5b's cores (`nice -n 19 taskset -c 5-7,13-15`), one
  heavy slot per ≤ 45-min chunk (`mem_cap.sh 12`).
- **Elapsed:** 2026-10-22 17:07 → 10-23 01:30 PT.
- **Load:** load1 mean 5.4 (max 9.9), beside D1 on the GPU.

## FINDINGS

1. **F-X52-R1: UNDECIDED.** The practised gain (+3.2) sits between "no familiarity" (UB < 4) and "familiarity shown"
   (LB > 0).
   - Resolving it would take ≈ 2× the games per cell, or a third seed.
   - Neither is registered. Any further read is a new registration.
2. **F-X52-R2: "@5M" means `final_model.zip`.** It is not `checkpoint_285000xxx_steps.zip`: the two differ by one PPO
   update (s3501: 28,950 vs 28,940 updates). The registration said "@5M" without naming the file. The X35 kit's
   "final" convention decided it.
3. **F-X52-R3: P\*'s per-half scores (0.522 / 0.558) are above the registered ≈ 0.50.**
   - They come from REVERSE cells, so they are no goal (b) number.
   - The unpractised half scoring higher than the practised one under P\* (both are unfamiliar to P\*) is
     between-half variation at 600 games: a 3.6 pp difference, n.d. by Wilson.
4. **F-X52-R4: The C0 drift cell reads −2.3 pp** on the unpractised half (n.d.). Continuation alone does not move it,
   so the descriptive X52 − C0 contrast (+4.0 on the same units) is the cleaner, unregistered view of
   transfer. It is not part of any rule.
5. **No other hazard hit.**
   - The halt marker was clear before every unit. Our server bound 9552 (never 8000 / 8001).
   - Nothing was written under `models/` or `data/`. No GPU. Metamon was a local peer only.

## Files

- `cells.json`: every cell, `analysis` (Δ per half, per seed, the descriptors, the class, uncounted shares, twins),
  every unit's counting verdict, the declared digests and the model sha256s.
- `rows.jsonl`: every counted game row, plus `read_cell`, `unit`, `unit_seed`, `their_team` and `our_team_stem`.
- `scripts/x52q.py`: the resumable one-lane queue. `scripts/aggregate.py`: the counting, capture check, statistic and
  classes. `scripts/make_configs.py` and `configs/`: the two reverse-cell configs, sha256 in `x52q.py`.
- Not committed: the raw unit dirs, captures, logs, `units.jsonl` and the read-only model copies, all in
  `~/gen3ai_archive/x52_read/`.
