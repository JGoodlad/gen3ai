# STATE-level RND novelty reads (2026-09-30) — does the RND input pick out unseen STATES?

**The question (owner, X25).** The ride-along RND (`RndNovelty`, random network distillation) reads the
raw observation. Meter (v) (`../smoke_2026-09-30/README.md` §(v)) chose it over the trunk features on a
TEAM-level proxy that was confounded with time. Does the choice hold at the level of STATES?

**Verdict: the observation-vs-features choice HOLDS at the state level. obs-RND (on the observation) is
equal to or better than feature-RND (on the trunk features) on every read. But the state-level signal is
MODERATE for both, and NEITHER variant picks out the successors of the starved near-best moves.**
- (2) Visitation: novelty falls with how often a state was seen, with Spearman ρ ≈ −0.34 for obs-RND
  and −0.31 for feature-RND.
- (3) Off-distribution states: genuinely different states read novel. Opponent class: 0.75 (obs) vs
  0.64–0.69 (features). Late game: 0.83 for both.
- (1) Same distribution: obs-RND MEMORISES battles. A held-out battle reads 0.85 vs train rows (features
  0.62), so most of obs-RND's novelty separates whole trajectories, not states.
- (4) Starvation: one ply away, the moves the policy starves lead to states that are no rarer than its
  other neglected moves. RND adds nothing toward anti-starvation. The X4 pre-read found the critic is
  just as blind (`../../x4_preread/READOUT.md`).

**Tag: MEASUREMENT on the fixed bank, offline predictors with the heads' own shapes; one feature
checkpoint (N0 final); S = 1 successor seed.**

## Setup

- **Bank:** `m5_laneS/bank_v1`: 20,712 decisions, 580 battles. It is re-encoded byte-equal to the
  recording (19,964 / 19,964 checkable rows).
- **Split:** battles split 70 / 30 WITHIN each (source × opponent class) cell, seed 20260930.
  - TRAIN: 406 battles, 14,024 rows. HELD-OUT: 174 battles, 6,688 rows.
  - Both sides have the same teams, eras and opponent mix, which removes the time confound of meter (v).
- **Predictors:** the heads' `RndNovelty` shapes (frozen target 2761→256→64, predictor →256→256→64,
  ±5-clipped normalisation fitted on the predictor's own training rows).
  - Adam at 3e-4, minibatch 256, seed 20260930, read after 3 / 10 / 30 epochs. **The headline is 10.**
  - obs-RND reads the 2761-dim observation.
  - feature-RND reads the 128-dim detached `value_pooled` of ONE checkpoint,
    `ai_v14_01_base/final_model.zip` (N0 final), so this read has no representation drift.
- **Intervals:** every interval is a 95 % percentile bootstrap that resamples BATTLES, with B = 1000 and
  seed 20260930. AUROCs use `boot.py`'s weighted mid-ranks; means use `qhat_report.Boot`.

## The numbers (10 epochs; the 3 / 30-epoch rows are in the JSONs and move nothing below)

| read | obs-RND | feature-RND |
|---|---|---|
| **(1)** AUROC, held-out rows vs TRAIN rows (≈ 0.5 = generalises) | **0.853 [0.831, 0.872]** (3 / 30 ep: 0.74 / 0.90) | **0.616 [0.563, 0.668]** (0.55 / 0.67) |
| **(2)** Spearman ρ of error vs log(1 + train count of the state key), held-out | **−0.338 [−0.409, −0.259]** | **−0.305 [−0.398, −0.202]** |
| (2) AUROC, key never seen in train (count 0) vs seen | 0.680 [0.647, 0.711] | 0.646 [0.600, 0.688] |
| (2) … the same, inside turn quintiles (controls for "rare = late") | 0.680 [0.642, 0.708] | 0.644 [0.600, 0.685] |
| (2) monotone steps, tie-free count bins (0, 1, 2–3, 4–7, … 64–127) | 5 of 7 | 3 of 7 |
| (2) monotone steps, 10 equal-count deciles of log(1 + count) | 7 of 9 | 7 of 9 |
| **(3)** trained on BOT games only: self-play vs bot | 0.747 [0.700, 0.790] | 0.642 [0.541, 0.732] |
| (3) trained on SELF-PLAY games only: bot vs self-play | 0.750 [0.694, 0.805] | 0.691 [0.605, 0.780] |
| (3) reference, trained on both classes: self-play vs bot | 0.548 [0.485, 0.606] | 0.480 [0.377, 0.580] |
| (3) trained on bot games only: exploiter vs bot | 0.719 [0.613, 0.809] | 0.649 [0.515, 0.766] |
| (3) late turn (turn > 85, the train p95) vs the rest | 0.828 [0.776, 0.890] | 0.827 [0.617, 0.948] |
| (3) both sides ≤ 2 mons alive vs the rest | 0.806 [0.749, 0.852] | 0.837 [0.757, 0.894] |
| (3) off-pool teams | **none exist** (below) | **none exist** |
| **(4)** AUROC, starved near-best successors vs argmax successors, held-out-battle turns (n = 110) | 0.519 [0.482, 0.559] | 0.447 [0.402, 0.492] |
| (4) within-turn paired log(error ratio), starved near-best vs argmax | **+0.068 [+0.018, +0.118]** | −0.001 [−0.050, +0.048] |
| (4) the same for the starved FAR moves (π < 1 %, gap > ε) vs argmax | +0.054 [+0.006, +0.104] | +0.021 [−0.019, +0.064] |
| (4) starved near-best − starved far, paired log ratio (n = 95) | +0.001 [−0.037, +0.040] | −0.037 [−0.080, +0.006] |
| (4) share of an RND error's variance that lies within a turn (between its one-ply actions) | 0.13 | 0.13 |

### Reading each row

**(1) obs-RND memorises.** With no time or team confound, a held-out battle reads 0.85 against train
rows. Meter (v) measured 0.86 for its held-out seen-team rows, so that number was almost entirely
memorisation, not era shift.
- The observation carries both full team blocks and the pair history. That makes every battle's
  trajectory close to a fingerprint, and the 790k-parameter predictor learns the fingerprints.
- feature-RND compresses that away (0.62). It generalises more, and it also discriminates less
  everywhere below.

**(2) Visitation: both variants track the count, moderately.**
- ρ ≈ −0.3 for both, and the error falls with the count across 5 of 7 tie-free bins for obs-RND (3 of 7
  for feature-RND).
- Each signal survives inside turn quintiles unchanged (0.68 / 0.64), so it is not just "rare = late".
- The state key: active species × active species, turn bucket, both active HP buckets, weather, spikes
  each side. It is decoded from the observation through `get_layout()`, and it agrees with the bank's
  own turn and alive stamps on 20,712 / 20,712 rows.
- 3,657 of the 6,688 held-out rows (55 %) carry a key never seen in train. 7,702 distinct keys occur in
  train.

**(3) Off-distribution: both variants flag it.**
- A predictor trained on one opponent class flags the other at ~0.75 (obs). The all-class predictor
  reads the same contrast at ~0.5, so the signal is the class the predictor did not see, not an intrinsic
  difference in error.
- Late-game rare configurations read ~0.83 for both variants. feature-RND's intervals are 2–3× wider.
- **Off-pool:** every team on both sides of all 580 bank battles is a training-pool team (764 team files
  under `data/teams`, the `default_biased` trainee builder's full pool). The 8 untaught-meter teams are
  pool files themselves: "untaught" means a teacher did not teach them, not that the trainee never saw
  them. There are NO off-pool states to read (FINDING).

**(4) Starvation: no state-level link that RND can see.**
- Coverage: N0 final's 951 decisive truth turns. 331 hold a starved near-best move (π < 1 %, truth gap
  ≤ ε = 0.1), 440 such moves in all.
- Every legal action was branched one ply: S = 1 (the truth's first seed); the opponent's open root
  decision (863 turns) and any later opponent-only decision were answered by N0 final's greedy choice,
  the X4 pre-read's variant M. That gave 7,042 branches, 21 of which ended the game; 0 turns were
  refused.
- **The headline is the held-out-battle turns.** On a TRAIN battle, the argmax successor is close to the
  recorded next state, which the predictor trained on. That alone makes every other move's successor
  look rarer: all turns read obs 0.56 [0.54, 0.59], and the plain state-key count shows the same
  self-inclusion.
- On held-out battles:
  - obs-RND reads every NON-played successor ~5–7 % more novel than the played one. This is detected in
    the paired ratio, but the AUROC is 0.52.
  - It reads the starved GOOD moves exactly like the starved BAD ones (near − far +0.001
    [−0.037, +0.040]).
  - feature-RND sees nothing.
- The state-key count agrees on held-out battles: starved near-best successors are not detectably rarer
  (AUROC 0.51 [0.46, 0.56]; Δ log(1 + count) +0.09 [−0.07, +0.24]).
- 87 % of RND's variance across these successors lies BETWEEN roots, so one ply is far below its
  resolution.
- An RND bonus would push toward unvisited one-ply successors in general. It would not push toward the
  near-best ones the policy starves.

## Command

```bash
export PYTHONPATH=$PYTHONPATH:src OMP_NUM_THREADS=4
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 nice -n 19 \
  python3 -m main.ridealong_read.rnd_states \
  --out designs/research_state/measurements/ridealong_baseline/rnd_states_2026-09-30 \
  --cache <scratch>/rnd_states_cache --archive ~/gen3ai_archive/ridealong_read/rnd_states --threads 4
```

- **Prerequisite:** read (4) needs this checkout's release build of the rust env library:
  `(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo build --release --lib --bin rust_env_proc)`.
- **Resumable:** the re-encoded rows, the N0 forward and the 18 predictor snapshots are cached, the
  successors are written in 50-turn chunks, and each read skips if its JSON exists.
- **Wall time:** about 10 min on CPU (4 threads), including ~4.5 min of successor branching.

## Files

- `read1_seen_vs_heldout.json`, `read2_visitation.json`, `read3_off_distribution.json` (including
  `off_pool`), `read4_starvation.json` (including `key_count_reference`).
- Every JSON carries the split, the training losses, the feature checkpoint's sha256, the re-encode gate,
  the bootstrap method and the reader commit.
- Per-row arrays: `~/gen3ai_archive/ridealong_read/rnd_states/`. This holds `bank_rnd_errors.npz`,
  `state_keys.npz`, `successor_rnd_errors.npz` and `successors_N0final_S1/chunk_*.npz` (the successor
  observations themselves). It is never committed.
- Code: `src/main/ridealong_read/rnd_states.py`. Tests: `rnd_states_test.py`.

## Caveats

- **One seed per successor.** The X4 pre-read found S = 1 reads like S = 32 for V. RND reads the
  observation directly, so it has no reason to differ. NOT RE-CHECKED here.
- **Decisive-turn count.** 951 decisive turns here, against the X4 readout's 943 for N0. This read uses
  `meters.truth_turns` over all 64 truth seeds; the readout's own counting differs. 331 of 951 decisive
  turns (35 %) hold a starved near-best move, against Lane S's "about half". The policy's π here is N0
  final's own forward on the re-encoded row. The gap is NOT RECONCILED.
- **Held-out-battle (4) turns are few.** Only 110 turns, so every (4) interval there is wide.
- **The offline fit is not the live learner.** The ride-along learner sees each rollout ~n_epochs times
  on a moving stream. The memorisation in (1) is the offline 10-epoch fit over 14k rows, and its size
  in the live learner is **UNVERIFIED**.
