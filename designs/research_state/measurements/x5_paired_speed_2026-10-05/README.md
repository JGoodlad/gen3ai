# X5 paired ABAB speed benchmark: s for Amendment 5 (2026-10-05)

The PAIRED benchmark Amendment 1 requires (`design_x5_belief_tokens.md` §7.5, read by §7.4's reader rule) and
Amendment 5 gates on (§7.9: fixed_mass launches at s ≤ 50 %). Under the GPU lease (owner `x5-perf-encode`), at
the worktree's tree = `065d2d7f` + the F-XC-3 fix (`../x5_launchable_2026-10-05/`), i.e. P_x5's code. Box: RTX
3080 Ti, torch 2.8.0+cu126, desktop STOPPED (T23), no other GPU job. 20:44–22:38 PDT. Tags: **MEASURED**.

## Verdict

| | blob | fixed_mass |
|---|---|---|
| median update-cycle wall, pooled (n kept) | **54.43 s** (21) | **63.53 s** (21) |
| block medians (A1 B1 A2 B2 A3 B3) | 54.454 · 54.434 · 54.332 | 63.777 · 63.377 · 63.516 |
| `train_ms` block medians | 40.96 · 41.00 · 40.92 s | 46.29 · 46.13 · 46.17 s |

- **s = +16.7 %** (pooled medians); per block pair **+17.1 %, +16.4 %, +16.9 %** (range 16.4–17.1 %).
- **s ≤ 50 %: the Amendment 5 launch condition HOLDS.**
- **Matched wall-time:** 15M / (1 + s) = 12.85M steps, so the **12M** checkpoint (the nearest 1M at or below).
  Deterministic over the block range: 15M / 1.171 = 12.81M and 15M / 1.164 = 12.88M both floor to 12M; the floor
  moves only at s ≥ 25.0 % (to 11M) or s ≤ 15.4 % (to 13M).
- Of the +9.1 s per cycle, the update (`train_ms`) is +5.2 s (+12.7 %, matching the gather unit's +12.8 %); the
  rest (≈ +3.9 s) is the rollout, where T2's X5 cost lives (`../x5_hyp_gather_2026-10-05/` F-HG-2).

## Design

`../x5_launchable_2026-10-05/scripts/queue3.sh` step 3; each block one fresh launch through `drive.sh`:
`--arch production --device cuda --steps 15000000 --seed 1001`, the X26 ride-along heads (`--ridealong-ensemble 5
--ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all`), `--snapshot-ladder-games 0
--checkpoint-every-steps 1000000`; fixed_mass adds `--belief-tokens fixed_mass --allow-nonproduction-arch`.
Regime A: each block's run dir is seeded with a fresh 20-snapshot pool of its own arm (`build_pool.py`). Eval at
its default (2M), so no eval cycle falls inside 10 updates. Order blob, fm, blob, fm, blob, fm. Stopped by SIGTERM to
the trainer's PID after 10 `update` phase rows (the startup dry update + 9 real updates). Only `PROF_PHASE_LOG` and
`PROF_LOG_SLOT_LOADS` (logging) were set; no override.

## The reader (`scripts/s_read.py`)

§7.4: an update-cycle is the wall between consecutive `train/` update records (`train/train_ms` in the run's own
TensorBoard; the dry update writes none). Excluded: a window holding an eval cycle (none occurred), a restart (one
TB run dir per block, asserted), the compile-canary update and the one after it (update 10 / 11; not reached in a
9-update block), any window with a CPU-sampler row at windowed contention ≥ 1.05 (the quiet rule; none fired),
and **the stop**: the record written after the driver's SIGTERM is the abort path's dump, whose window holds an
update but no rollout (43–48 s against 54–64 s). The reader as first written kept that cycle; it is excluded here
(`scripts/s_read_before_stop_fix.py` is the earlier version: it read s = +17.0 % on pair 1, the fixed one +17.1 %).
s = median(fixed_mass, pooled) / median(blob, pooled) − 1.

Every block's per-cycle walls and exclusions: `results/s_read.json`. Per block (`results/blocks/`): the phase
markers (every slot load's growth), the driver log, and the startup key lines (T2, R1 gate, stage ledger,
`UpdateFit`).

## Also read off the same six launches

| | blob ×3 | fixed_mass ×3 |
|---|---|---|
| T2 up | 243–245 s | 495–496 s |
| startup slot loads (20 each) | max growth 1,024 B, 0 refused | max growth 1,024 B, 0 refused |
| T2 stage allocated | 731 MiB (`blobB` before the F-XC-3 fix: 920) | 749 MiB |
| `UpdateFit` headroom (declared 1,024) | 3,080–3,082 MiB (`blobB`: 2,874) | 1,790–1,798 MiB |

Blob's T2 allocation falls 189 MiB with the F-XC-3 fix (its replicas no longer keep a gate's stash); its
compiled code and outputs are unchanged (`../x5_launchable_2026-10-05/` §2).
