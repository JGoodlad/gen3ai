# The M5 SWITCH — `--env-core rust` as the production env core (`gen3_env_core_switch_v1`)

**Status: SHIPPED 2026-10-02 (ledger *THE M5 SWITCH*) at the pre-sizing N = 48 shape.** The orchestrator
released it before the M5 SIZING verdict; the verdict sets N\* and the verdict-dependent `recipe.sizing` rows
(`n_envs`, `n_steps`, `rollout_target_samples`; `n_epochs` in `recipe.fresh`) in its own commit
(`program_rust_core.md` M5, "THE SWITCH"; order constraint 5).

## What the flip is

- `designs/production_config.json` `recipe.sizing` — the ONE block the sizing verdict fills: `env_core`
  (rust), `n_envs` (48 until N*), `n_steps` (2048, the maximum), `rollout_target_samples`, `trainee_slots`,
  `t2_buckets`, `t2_lanes` (null = the collector derives them), `verdict` (null = PENDING).
- `--arch production` applies it like every recipe knob (`main.train.recipe_surface`); the collector-only
  rows are applied and compared only on the Rust core.
- An untyped `--env-core` (`main.train.rust_env_setup.resolve_env_core_default`, called by
  `resolve_config` and `checkargs`): fresh `--arch production` → rust · `--model` (a restart or a fork)
  → INHERITED, the core the checkpoint was produced on (python when recorded before `--env-core`) · a bare non-production
  fresh argv → python (it defaults to `--critic shaped`, which the Rust core refuses; decided in the
  deletion pass — manifest row in `program_rust_core.md` §4).
- `--env-core python` (typed) stays reachable until the deletion pass; reported as a TYPED deviation.
- `--dry-run` / `checkargs` / the launcher print `env core : rust [production core, recipe.sizing]` and a
  `sizing : … verdict PENDING` line.
- Pinned by `src/main/train/env_core_switch_test.py`.

## Pre-flight at N = 48 (today's sizing), under the launcher

PRE-FLIGHT-ONLY overrides (not part of the flip): `--eval-freq 100000` (so the first eval cycle lands
inside a ≤ 20-min GPU hold) and, from #2, `--supply-starve-cycles self_play_pool=0` and
`--checkpoint-every-steps 150000` (the shortened eval cadence trips the self-play supply guard after 3
cycles, and a checkpoint is needed for the resume). The runs were made in the prep worktree's own `models/`
(never `models/` of the main checkout) and are kept, with the drivers and launcher logs, in
`~/gen3ai_archive/m5_switch/` (`m5sw_preflight`, `m5sw_preflight2`, `m5sw_preflight256`).

### #1 — fresh `--arch production --env-core rust` (2026-10-01 11:26–11:37, pin `df8a42ea`)

`--steps 300000 --device cuda --eval-freq 100000`, gpu_lock + mem_cap 48 GB, `timeout 1200` inside.

| | |
|---|---|
| launcher start → child | 11:26:40 |
| env core build | 6.5 s (`🦀 [ENV CORE BUILD]`, release, fresh pin worktree) |
| T2 up | 161.3 s — 1 slot group (`pool0`, 31 slots), trainee slot 24, buckets (8, 48), 8 lanes, graph on cuda |
| eval core up | 0.1 s, 64 envs, 9 bots in core + 5 sentinel slots |
| learner compile gate + prewarm | 11:29:40 → 11:34:27 (R0 B=48, R1 B=2048; 6 graphs, 1 entry per code object) |
| **startup → compile LOCK** | **7.8 min** |
| updates | 3 × 98,304 rows; `update_wall_s` 46.9 / 41.1 s; `compile/recompiles_after_lock` 0; `lifecycle/eager_share` 0; 480 compiled-region calls per update |
| rollout | `rust_env/trainee_decisions_per_s` 11.5k (first) → 4.8k / 4.4k; core 1.6–2.2 ms and GPU wait ~1.7 ms per host step |
| eval | 3 cycles at 100k / 200k / 300k: bots 4.9 % → 10.2 % → 21.1 % |
| CUDA ledger (`lifecycle/cuda_*`) | not reported — it logs at its window's close, after update 3 |
| compile canary | armed, cadence 100 updates — not reached |
| end | **FATAL_SUPPLY (5)**: the self-play pool still empty after 3 eval cycles (the guard under the shortened eval cadence, not the core); compile lock released with 0 compiles / 0 rejected after it; peak RSS 14.9 GB; no checkpoint written |

### #2 — fresh with a checkpoint, then the `--model` same-run resume

Pin `e6b9cc2f` (the FLIP), launcher, gpu_lock + mem_cap 48 GB, `timeout 1200` per hold, overrides
`--eval-freq 100000 --supply-starve-cycles self_play_pool=0 --checkpoint-every-steps 150000`.

**Fresh** (16:34–16:54): `--arch production --steps 300000 --run-name m5sw_preflight2`, NO `--env-core`.

| | |
|---|---|
| core chosen | `RustVecEnv` by the flipped default (metadata `env_core` rust, `cli_args.env_core` rust) |
| T2 up | 216.5 s |
| **startup → compile LOCK** | **7.9 min** |
| updates | 3, 0 compiles after the lock; checkpoints 150k / 300k |
| eval | bots 4.9 % → 10.2 % → 20.4 % |
| CUDA ledger (MiB, alloc / reserved) | weights 15 / 34; Rust core +712 → 728 / 1,714 (peak 1,111); compiled regions gate + prewarm 926 / 5,894 (peak 5,034); optimizer state 950 / 5,894; update peak alloc 8,760, reserved 9,490 |
| compile canary | armed "at update 10, then every 100" — not reached in 3 updates |
| end | exit 124: the hold's `timeout` during the in-process FINAL EVAL (training had finished; vs random 93 %, vs heuristic 25 % before it). The final eval is OFF by default since `e6121412` |

**Resume** (17:13–17:33): `--model checkpoint_300000 --steps 500000`, no `--arch`, no `--env-core`.

| | |
|---|---|
| core chosen | the recipe restart route restored `env_core='rust'` (+ every sizing / recipe row) from `cli_args` |
| compile cache | REUSED (1.4 GB) |
| T2 up | 42.4 s |
| **startup → compile LOCK** | **1.8 min** |
| updates | 2, `update_wall_s` 40.7 / 40.6; checkpoint 450k; `final_model` saved; exit 0; peak RSS 8.7 GB |
| eval | 19.9 % → 28.2 % |
| CUDA ledger (MiB) | weights 39 / 72; Rust core 751 / 1,418; regions 950 / 5,684 (peak 5,022); update peak alloc 8,750, reserved 9,400; `device_batch_mib` 1,140 (the resident micro-batch copy, before `7ef99979` made `--device-batch staged` the default) |

No K9(b) behaviour-check FATAL in either. The launcher printed `⚠️ launcher render error: cannot convert
float NaN to integer` once (16:43:50, fresh, headless) — harmless, not investigated.

### #3 — N = 256, production + the X26 ride-along heads, fresh then resume (after `7ef99979`)

Pin `55f601ff` (the switch rebased on `7ef99979`'s memory fix), launcher, gpu_lock + mem_cap 48 GB,
`timeout 1200` per hold, the #2 overrides. Run dir `~/gen3ai_archive/m5_switch/m5sw_preflight256` (drivers `preflight256.sh` / `resume256.sh` beside it).
Argv: `--arch production --n-envs 256 --n-steps 384 --ridealong-ensemble 5 --ridealong-rnd
--ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all --steps 294912`, NO `--env-core`.
**This is the configuration's PRE-FLIGHT, not a sizing claim: production stays at N = 48 until the
SIZING verdict and the steady-state memory fix (below).**

**Fresh** (2026-10-01 23:13–23:25, after sizing arm B):

| | |
|---|---|
| core chosen | `RustVecEnv` by the flipped default; the X26 heads ON (6 ride-along optimizers acquired at the freeze) |
| env core build | 9.0 s |
| T2 up | 241.9 s — 1 slot group (`pool0`, 31 slots), trainee slot 24, buckets (8, 64, 256) (non-trainee slots ≤ 64), 8 lanes |
| **startup → compile LOCK** | **7.8 min** (23:13:19 → 23:21:09) |
| `update_fit` (startup) | demand 7,846 MiB reserved, peak allocated 6,426, floor 1,128; card for the process 10,530 → **headroom 2,684 MiB** (bar 1,024); 13.6 s |
| real first update | peak allocated **6,230** (dry 6,426), peak reserved **7,912** (dry 7,846: +66) |
| reserved per rollout / update (`lifecycle/cuda_{rollout,update}_peak_reserved_mib`) | 7,846 → 7,912 → 7,978 MiB — **+66 MiB per update**, the slope sizing arm B showed to its plateau near update 33 (where B, heads OFF, read 341.5 MiB of headroom: D-6 FAILS at steady state — owned by the next memory unit) |
| updates | 3 (the third's scalars are never dumped at `learn()`'s end); `update_wall_s` 49.6 / 42.3; `compile/recompiles_after_lock` 0; `lifecycle/eager_share` 0 |
| rollout | 103,424 / 98,560 / 98,816 completed-game rows; `rust_env/trainee_decisions_per_s` 2,954 → 3,680 → 4,908 |
| eval | bots 0.5 % → 5.6 % → 11.2 % |
| CUDA ledger (MiB, alloc / reserved) | weights + heads 56 / 66; Rust core 905 / 2,180 (peak 1,295); compiled regions 1,104 / 6,300 (peak 5,223); optimizer state 1,128 / 6,300 |
| end | `Training complete`, exit 0; 0 compiles / 0 rejected after the lock; freeze released, 6 checks passed; no K9(b) FATAL; peak RSS 15.3 GB |

**Resume** (2026-10-02 00:11–00:16, after sizing arm C): `--model checkpoint_300032 --steps 491520`, no
`--arch`, no `--env-core`, no head flags.

| | |
|---|---|
| core chosen | the recipe restart route restored `env_core='rust'`, `n_envs=256` and every recipe row from `cli_args`; the heads inherited (6 optimizers) |
| compile cache | REUSED (1,662.6 MB) |
| T2 up | 38.9 s |
| **startup → compile LOCK** | **1.8 min** |
| `update_fit` | demand 7,712, **headroom 2,826 MiB** |
| real update | peak allocated 6,230, reserved 7,778 (dry 7,712: +66) |
| updates | 2, `update_wall_s` 49.4; 0 compiles after the lock |
| eval | bots 11.1 % → 13.9 % |
| end | exit 0; peak RSS 10.1 GB |

**Found by it, fixed in the switch commit:** `checkargs` and `launcher --dry-run` REFUSED this resume
("--env-core rust requires --critic winprob") while the launch itself ran it. `checkargs` resolved
`--critic` to its parser default (`shaped`) before reading the parent's `model_config.json`; the launch
inherits `winprob`. The switch exposed it (an untyped `--env-core` on `--model` now inherits `rust`), and
so did any resume argv that TYPES `--env-core rust`. `checkargs` now resolves the critic against the
parent in the launch's order (`checkargs_test.py::test_a_resume_of_a_winprob_rust_run_inherits_its_critic_and_is_not_refused`,
fails on revert).

## Ledger

`ledger.md` 2026-10-02 *THE M5 SWITCH* (ERA BOUNDARY).
