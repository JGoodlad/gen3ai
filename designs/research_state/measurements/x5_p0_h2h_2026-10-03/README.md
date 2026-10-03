# X5 P0: the run floor σ_h on the HEAD-TO-HEAD scale (2026-10-03)

**What this measures.** The run-to-run SD of the mirrored head-to-head meter, σ_h, on the four banked M5 sizing
replicates' FINAL snapshots — the number `designs/endstate/design_x5_belief_tokens.md` §7.4's power simulation borrows
from the untaught meter (3.43). **A PLANNING input, not a meter selection** (§7.3). 120,000 games, CPU / GPU, nothing under
`models/` written; every checkpoint READ-ONLY. The tool is `python -m main.h2h` (`designs/training/eval_and_rating.md` "The
checkpoint-vs-checkpoint head-to-head"); the rows are its §0b COUNT rows (`gen3_eval_count_row_v1`).

**Headline (evidence tags: MEASURED here; the power rows are the registered design's own simulator run at the measured σ).**
- σ_h of the runs that match the A/B's conditions (one commit; S2 below) is **0.33 pp** (2 df, 95 % one-sided upper bound
  **1.79**); of the pure seed pair (S1) **0.53 pp** (1 df). The conservative set that includes a run on the other side of a
  declared controller-regime boundary (S3) gives **2.12 pp** (2 df; 95 % CI [1.09, 13.4]); with B's N lever as well (S4)
  **1.63 pp** (3 df; CI [0.91, 6.12]). The untaught meter's borrowed value is **3.43** (1 df): the head-to-head's run floor is
  SMALLER than the planning number, by 1.6x (S3) to 10x (S2).
- **Power at δ = 3.5 pp (registered design, cell SE 1.05 pp measured):** 0.57 at the borrowed 3.43 → **0.905 at S3's 2.12**
  (0.984 at δ = 4.5), 0.965 even at S2's upper bound 1.79, ~1.00 at the S1 / S2 points. At the 8M depth the δ = 3.5 vs 4.5
  question is moot; what the data CANNOT exclude (2-3 df) is a σ_h at 15M above ~3: at S4's upper bound 4.79 power is 0.36.
- **The untaught meter is the wrong yardstick for run strength at this scale:** A2 − A′ is +4.90 pp on U (1.8 % meter share)
  and **−0.82 ± 0.33 pp** head to head — the opposite sign. The six head-to-head differences regress on the same runs' U
  differences with slope 0.26 (5 dof, six differences sharing four runs).
- **Seat effect u = −0.006 ± 0.083 pp** (self-play and reversed edges agree): the player keeping seat p1 biases nothing.

## Files

| file | what |
|---|---|
| `rows/ledger.*.jsonl` | the 120 §0b COUNT rows (12 edges × 10 batches of 500 mirrored pairs), validated by `python -m main.h2h read rows` |
| `speed_probe_cpu/`, `speed_probe_gpu/` | the SAME 100 pairs of A2 vs A′ on CPU eager and on GPU graph (throughput, and the cross-device equality below) |
| `logs/` | each edge's `[h2h]` lines (engine start, per-batch games/s, W/L/D, pentanomial) and the memory-cap / lock lines |
| `provenance.json`, `scripts/provenance.py` | the four runs: pins, seeds, N, effective hyperparameters, final snapshot + sha256, the config comparison |
| `scripts/run_edges.sh` | the driver (resumable: `main.h2h play` skips banked batches) |
| `scripts/analyze.py` | edges → seat effect → σ_h (`main.h2h.runfloor`) → U comparison → the power simulation; `--check` is the readout gate's |
| `result.json`, `result.md` | its output (generated; `result.md` carries every table) |

Reproduce: `PYTHONPATH=<checkout>/src python3 scripts/provenance.py && scripts/run_edges.sh && python3 scripts/analyze.py`
(the interpreter is `gen3ai_torch28`; `run_edges.sh` needs the GPU lock and ~46 min).

## The four runs: which are replicates, and exactly what differs

`models/sizing_*`, final snapshot = `resolve_model_ref`'s last (`latest.txt`; `final_model.zip`, num_timesteps = the run's).
Same on ALL FOUR (verified, `provenance.json`): `model_config.json` (sha256), the pickled `policy_kwargs` of the final snapshots
(extractor and policy architecture, 16,350 flattened keys), torch 2.8.0+cu126, the matchup spec `ef5242cffd` (eval trainee
teams `default_biased`, bias 0.1), `--arch production`, and every EFFECTIVE hyperparameter read from the final zips — batch
2048, 10 epochs, lr 3e-4, gradient accumulation 32, γ 1, λ 0.8, ent 0.05, vf 0.5 — **except `n_steps`: 2048 for A / A2 / A′,
384 for B** (× n_envs: the same 98,304-sample rollout).

| run | pin | seed | N (envs) | final steps | differs from the others |
|---|---|---|---|---|---|
| **A2** `sizing_A2_n48_e10_s1001` | `277f318f` | 1001 | 48 | 8,130,046 | **crash-restart** at 4.52M from `checkpoint_4000032` (a K9(b) FATAL): 0.52M steps re-collected on a new trajectory; 0.84 % more steps |
| **A′** `sizing_Ap_n48_e10_s1002` | `277f318f` | **1002** | 48 | 8,062,352 | the seed |
| **A** `sizing_A_n48_e10_s1001` | **`f9349f95`** | 1001 (A2's id) | 48 | 8,062,355 | **the pin, BEFORE `f6b32f5a`** (the KL→LR controller lost 1 reading per eval cycle; the commit's own words: "a REGIME BOUNDARY for live-controller runs"). 12 commits; the others are not learning-relevant on this recipe (`design_x5` §7.1) |
| **B** `sizing_B_n256_e10_s1001` | `7ef99979` (20 commits after `277f318f`; contains `f6b32f5a`) | 1001 | **256** | 8,068,077 | **N = 256 / `n_steps` 384**, and a third pin |

- **A2 vs A′ is the only pure seed pair** (same pin, same N, same recipe). It carries A2's crash-restart.
- **A is NOT a replicate of the other three in the strict sense:** it is the only run on the pre-boundary side of
  `f6b32f5a`. A2, A′ and B are all post-boundary. With one run per side, a pin effect and a seed effect cannot be separated.
- **B differs by the N lever** (and its pin), not by seed alone: the note keeps it a sensitivity edge.
- A record artifact, harmless: A2's resumed segment recorded `cli_args.batch_size 4096` and `unified_damage/moves off`; its
  final zip's effective batch size is 2048 and its pickled `policy_kwargs` equal the others' (the recorded flags are not
  what the checkpoint carries).

## Method

- **Play.** Per edge, 5,000 mirrored team pairs (10,000 games) of the PLAYER (seat p1) against the OPPONENT on the Rust
  eval core, both sides GREEDY, teams from the player's eval builder (the default pool, 10 % sample-team bias), turn limit
  the training stall threshold. Regime id `191bb5810017ba85` on every row. 12 edges: the three replicate pairs in BOTH directions
  (the two directions of a pair draw the same team pairs, so `d_ij + d_ji = 2u`), B against each (one direction), and A, A2, A′ each
  against ITSELF (the seat effect). Seed 0; each edge's schedule key is the digest of the two checkpoints' hashes.
- **Meter SE: 0.46–0.48 pp per cross edge, 0.15–0.17 per self-play edge** (pair-clustered, unbiased; target ≤ 0.5 met). The
  cross edges' pair SD is 0.33, so a 1,000-pair cell has SE **1.05 pp** (the note assumes 1.1).
- **Scale.** `d_ij = 100 (score_ij − ½)` pp, and `d_ij = s_i − s_j + u + e_ij` (`main.h2h.runfloor` docstring: the model, the
  estimator `σ̂² = (Σ ŝ_i² − (2/K²) Σ_e v_e)/(K − 1)` with the meter's variance SUBTRACTED, and its exact chi-square interval). For
  K = 2 it is the note's per-edge formula `(d² − SE²)/2`.
- **The mapping to the untaught scale.** U reads a run's win rate against ONE fixed opponent on 8 pinned teams; the head-to-head
  reads it against another run on the trainee's team distribution. If strength were one number θ (Bradley-Terry), both
  meters' pp per unit θ is `P(1 − P)` (0.25 at ½, 0.225–0.248 over U's 34–46 %), so σ_h and σ_run(U) would be comparable
  number for number. **The data say they are not** (below): U's run differences are mostly NOT head-to-head strength.
- **Hardware / code.** Rows' `commit` = `a8e6e803` (the tool + the engine-start build fix); T2 `graph` on the RTX 3080 Ti under
  `scripts/ops/gpu_lock.sh` + `scripts/ops/mem_cap.sh` (20 GB cap, peak 5.2 GB), 64 envs, torch 2.8.0+cu126.

## Throughput (G-8)

- **GPU (T2 graph, 64 envs): 45–171 games/s per 1,000-game batch, median 130, pooled 118** (the box carried other agents'
  test gates throughout; the rows' `load_avg_start` reaches 17.7, contention factor ≤ 1.11). 10,000 games ≈ 85 s.
  **The engine start is 114–167 s per edge** (every edge builds its own T2 graphs): at 1,000 pairs per A/B cell the play
  (~17 s) is dwarfed by the start — a multi-cell engine (one architecture; weights loaded per cell, `svc.load`) is the
  follow-up that matters (FINDING F-P0-4).
- **CPU (T2 eager, 64 envs, 8 torch threads): 4.77 games/s** at load average 6–10 → **a 2,000-game cell ≈ 7 min**; the
  note's 64-cell look would be ~7.5 h on CPU against ~18 min of play on the GPU.
- **CPU eager and GPU graph play the SAME games:** the 100-pair probe on both devices gives identical W/L/D (101 / 99 / 0), the
  identical pentanomial `[19, 0, 61, 0, 20]` and the identical per-team counter map (174 teams) — `speed_probe_*`. Across the 120,000
  game rows 0.18 % of games contain a decision within 2e-5 of a tie (218 games), 0.68 % end in a draw (821).

## Results (every edge: 5,000 pairs; `result.md` has the pentanomials)

| player vs opponent | score % [95 % pair CI] | d (pp) |
|---|---|---|
| A2 vs A′ · A′ vs A2 | 49.29 [48.36, 50.21] · 50.92 [50.00, 51.84] | −0.71 · +0.92 |
| A vs A2 · A2 vs A | 47.52 [46.59, 48.44] · 52.47 [51.55, 53.39] | −2.48 · +2.47 |
| A vs A′ · A′ vs A | 45.33 [44.42, 46.24] · 54.47 [53.56, 55.38] | −4.67 · +4.47 |
| B vs A · B vs A2 · B vs A′ | 52.33 [51.39, 53.26] · 49.75 [48.84, 50.66] · 49.40 [48.47, 50.33] | +2.32 · −0.25 · −0.60 |
| A vs A · A2 vs A2 · A′ vs A′ (self) | 50.12 · 49.96 · 49.90 (each CI ±0.3) | +0.12 · −0.04 · −0.10 |

**Seat effect.** `u` = −0.007 ± 0.092 (self-play) and −0.003 ± 0.191 (reversals); pooled **−0.006 ± 0.083 pp**. The two directions
of every edge sum to ~0 (−0.01, −0.20, +0.21 pp). **A self-play pair is NOT exactly 0.5:** 5.2 % of self-play pairs are WW or LL
(264, 240, 277 of 5,000 per edge), and 779 of the 14,963 self-play pairs that are CLEAR of a near-tie decision are off centre — so it is
not rounding. On perturbed-fresh checkpoints the mirrored decision margins agree to 5 digits until a seat-dependent engine
event (a speed tie broken by the RNG in a seat-dependent order) and jump after it; that diagnosis was NOT repeated on these
trained checkpoints. The pair-clustered CI is the honest one.

**σ_h** (seat effect subtracted; meter variance subtracted; both directions of every pair; df = K − 1):

| set | what it is | strengths (pp, centred) | σ̂_h | 95 % CI | one-sided 95 % upper |
|---|---|---|---|---|---|
| **S1** {A2, A′} | the pure seed pair (1 df) | A2 −0.41, A′ +0.41 | **0.53** | [0.11, 18.45] | 9.22 |
| **S2** {A2, A′, B} | post-boundary pins, B's N inside (2 df) | A2 −0.19, A′ +0.47, B −0.28 | **0.33** | [0.00, 2.56] | **1.79** |
| **S3** {A, A2, A′} | seed + the pin boundary, all N = 48 (2 df) | A −2.35, A2 +0.55, A′ +1.80 | **2.12** | [1.09, 13.37] | 9.39 |
| **S4** {A, A2, A′, B} | everything (3 df) | A −2.34, A2 +0.48, A′ +1.50, B +0.37 | **1.63** | [0.91, 6.12] | 4.79 |

The note's per-edge recipe (the forward edge only, one direction) gives 0.37 (S1) and 2.13 (S3): the reversed edges add no
bias and only halve the meter variance, which was already 1–2 % of the spread. Largest |pairwise| difference: 4.57 pp (A vs A′).
**Which set to plan with.** S2 matches the A/B's conditions (one commit; N as a lever, as the A/B's N = 256 is B's); S3 is the
conservative one, because A sits across a declared regime boundary the A/B will not contain. With 1–3 df no interval excludes
a σ_h of 2–5; "the run floor is small" is the point estimate and the one-sided bounds (1.79 for S2), not a proof.

**Versus the untaught meter** (σ_run 3.43 for A′ − A2 at 1 df; 7.98 for A − A2; 2.97 / 5.21 pooled; 4.84 SD of the four E10 levels):

| pair | ΔU (pp) | head-to-head (pp) |
|---|---|---|
| A − A2 | −11.35 | −2.48 |
| A − A′ | −6.46 | −4.57 |
| A − B | −8.65 | −2.33 |
| A2 − A′ | **+4.90** | **−0.82** |
| A2 − B | +2.71 | +0.24 |
| A′ − B | −2.19 | +0.59 |

Slope of head-to-head on ΔU through the origin: **0.26** (naive SE 0.10; 5 dof, but six differences share four runs). The A
deficit is on both meters (so U is not noise); the +4.9 pp A2 − A′ gap on U is not on the head-to-head, consistent with the
note's G-A guard meter (U's run component is specific to its ONE opponent and 8 teams, and 98 % of A2 − A′'s gap is run-level,
not meter: 1.8 % meter share, §7.1). So **U's run floor does not transfer to this scale, and σ_h cannot be read off it.**

**A/B power** (`scripts/analyze.py`, the registered design verbatim: O'Brien–Fleming looks at 3 / 5 / 8 seeds, boundaries ×1.02,
futility stop on, 200,000 replications per cell, cell SE 1.05 pp; it reproduces the note's table within simulation noise: 0.806 / 0.572 at σ 2.5 / 3.43 against the log's 0.806 / 0.571):

| σ | source | power at δ = 3.5 | at δ = 4.5 | type-I (3.5) | E[GPU-h] (3.5) |
|---|---|---|---|---|---|
| 0.33 / 0.53 | S2 / S1 points | 1.000 | 1.000 | 0.024 / 0.033 | 18.7 / 20.3 |
| 1.79 | S2 one-sided upper | 0.965 | 0.997 | 0.045 | 30.1 |
| **2.12** | **S3 point** | **0.905** | **0.984** | 0.045 | 32.2 |
| 2.5 | the note's row | 0.806 | 0.944 | 0.045 | 33.6 |
| 3.43 | the borrowed U value | 0.572 | 0.763 | 0.046 | 34.5 |
| 4.79 | S4 one-sided upper | 0.364 | 0.515 | 0.047 | 33.6 |
| 9.39 | S3 one-sided upper | 0.158 | 0.211 | 0.047 | 30.6 |

## FINDINGS (standing rule 7)

- **F-P0-1 (the planning σ).** On this scale σ_h is far below the 3.43 the §7.4 simulation borrows: use **2.12 (S3, conservative)**,
  with S2's 0.33 (upper 1.79) as the A/B-matching evidence. Power at δ = 3.5 is then 0.90–1.00, not 0.57, and the §9 δ-vs-budget
  trade-off does not bind as long as σ_h at 15M stays at or below ~2.5. **UNVERIFIED at the A/B's depth (15M, N = 256, one commit):** no head-to-head σ_h exists there, and
  U gave no depth evidence either (3.43 at 8M, 2.41 at 75M, 1 df each); the first look's three seeds per arm will estimate it
  with 4 df. At small σ the registered boundaries are conservative (type-I 0.024–0.033) and the design stops early (E[GPU-h]
  ~19, not 34).
- **F-P0-2 (U is the wrong yardstick here).** A2 − A′ is +4.90 on U and −0.82 ± 0.33 head to head; the slope is 0.26. Anything
  that reads U differences as strength differences between replicates (the sizing study's and the learner battery's margins,
  `design_x5` §7.2's table) is on a scale roughly 4x too wide for the head-to-head (slope 0.26 ± 0.10).
- **F-P0-3 (A is a regime-boundary outlier, not demonstrably noise).** A is the only pre-`f6b32f5a` run and the only low one
  (−2.35 against post-boundary runs within ±0.5), on BOTH meters. One run per side cannot separate the pin from the seed. The
  note's "A − A2 is probably mostly run noise" (§7.1) is not supported at S2's point estimate (A at −2.35 would be a draw ~7σ out
  of a σ̂ of 0.33) and is not excluded at S2's upper bound (1.79: a 1.3σ draw); A2 / A′ / B agree within 0.8 and A vs A2 is
  −2.48 — consistent with a pin effect, which this design cannot confirm.
- **F-P0-4 (the tool's cost is the engine start).** 114–167 s of T2 graph compile per edge against ~17 s of play per 1,000-pair
  cell; the A/B's 9 / 25 / 64 cells need one engine held across cells (same architecture, `svc.load` per cell). NOT built (rule 9).
- **F-P0-5 (the mirror is not exact; the seat effect is 0).** See the seat paragraph. For the A/B's one-direction cells: u =
  −0.006 ± 0.083 pp at these runs, so `Δ̂ = mean h_ij − 50` carries no seat bias; a cheap self-play edge in the A/B pre-flight
  re-checks it.
- **F-P0-6 (§0b has no `purpose` for a pre-registered A/B read).** Rows are `audit`; the closed list's owner decides.
- **F-P0-7 (determinism, measured).** CPU eager and GPU graph agree exactly on 200 games of trained checkpoints; a re-run of one
  configuration is bit-identical; a change of env count changes no game clear of a near-tie (tests).
- **F-P0-8 (limits of the estimate).** The strengths of one run enter three edges, so K counts RUNS (df = K − 1), not edges; the two
  directions of an edge share a team schedule, which makes the combined meter variance slightly CONSERVATIVE (it is 1–2 % of the
  spread either way); normality of the strengths cannot be checked at 2–3 df; σ_h includes whatever snapshot jitter separates one
  final from another (the 8.0M-checkpoint jitter read of §7.3 item 2 was NOT part of this unit's brief and was not run).
