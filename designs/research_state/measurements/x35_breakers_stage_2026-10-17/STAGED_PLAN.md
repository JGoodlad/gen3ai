# X35, plateau-breakers: STAGED ON PAPER (2026-10-17)

**Status: STAGED, not launched; every launch NEEDS THE ORCHESTRATOR'S GO.** The spec is
[`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6. Its backlog rows are
`EXPERIMENT_BACKLOG.md` CE-3 to CE-3g, and its builds are `TASK_BACKLOG.md` T54–T59. X35 starts when X26 (`rb_x26_s2601`)
and X34 (`rb_toy_ent_s4002_deep`) have both reached their Tier-1 plateau and the GPU is free.

**How this was staged.** Everything ran at `9ebda190` with `CUDA_VISIBLE_DEVICES=""`, under `nice -n 19 taskset -c
5-7,13-15`, with no GPU lease. It used only `main.checkargs`, `main.launcher --dry-run` and one offline call to
`resolve_stable_opponents`. Nothing was created under `models/`, and nothing was written into X26's directory.

## 1. The stand-ins, and what replaces them at the plateau call

| stand-in here | at the call |
|---|---|
| P\* = `models/rb_x26_s2601/checkpoints/checkpoint_180000055_steps.zip` (the newest 10M node at staging) | the newest node of X26's second FLAT Tier-1 check, sha256 frozen |
| pin `9ebda1904da7a7782f22de2495b6e4c51368510b` | `X35_PIN_1`: a main commit carrying T54 (and T56 / T57 if they landed) |
| A1's archive = 20, 40, …, 160M (`a1_archive_spec_standin.txt`) | N_k = the 10M-multiple checkpoint nearest k·(P\* − 20M)/8, k = 1..8, ties down |
| `--steps 200000055` | P\* + 20,000,000 (production), T\* + 70,000,000 (toy) |
| toy C0 from `rb_toy_ent_s4002/final_model.zip` (100,081,428) | T\* in `rb_toy_ent_s4002_deep` |

## 2. The validated argvs (launcher flags: `--restart-interval-hours 24 --pin-commit <PIN>`)

| arm | argv file | `checkargs` | `launcher --dry-run` |
|---|---|---|---|
| **A1 s3501 (the FIRST arm)** | [`argv_a1_s3501_standin.txt`](argv_a1_s3501_standin.txt) | exit 0 ([`checkargs_a1_9ebda190.txt`](checkargs_a1_9ebda190.txt)) | "would launch", FORK of `rb_x26_s2601`, +20,000,000 ([`dryrun_a1_9ebda190.txt`](dryrun_a1_9ebda190.txt)) |
| C0 s3501 | [`argv_c0_s3501_standin.txt`](argv_c0_s3501_standin.txt) | exit 0 | "would launch", +20,000,000 |
| A2 s3501 | [`argv_a2_s3501_standin.txt`](argv_a2_s3501_standin.txt) | — | "would launch", +20,000,000 |
| A3 phase 1 (main exploiter) | [`argv_a3_main_exploiter_standin.txt`](argv_a3_main_exploiter_standin.txt) | — | "would launch", +5,000,000. **Without `--no-self-play` it is REFUSED** (F16) |
| A6 phase 1 (league exploiter) | [`argv_a6_league_exploiter_standin.txt`](argv_a6_league_exploiter_standin.txt) | — | "would launch", +5,000,000 |
| toy C0 | [`argv_toy_c0_standin.txt`](argv_toy_c0_standin.txt) | — | "would launch", FORK of `rb_toy_ent_s4002`, +70,000,000 |

**A1's archive resolves.** `resolve_stable_opponents` was called against X26's `ModelVersion`, offline
([`a1_archive_resolution_standin.txt`](a1_archive_resolution_standin.txt)). It returned 8 entries, rung
`explicit_zip`, with 8 unique labels (`ext_x26_020m` … `ext_x26_160m`). The arch gate passed. The dry-run cannot show
this, because the stable-opponent resolver runs in the child.

**What the dry-runs show for every production fork:**
- the 12 recipe knobs RESTORED from X26's record and passed typed (`n_envs=256 … n_epochs=10 … ent_coef=0.05
  self_play=True`);
- `--lr` / `--batch-size` / `--n-steps` INERT;
- the ARCH SURFACE matching the production mirror;
- a bf16 trunk;
- 20.68 GiB of disk REQUIRED against ≈ 318 GiB free;
- the Showdown deps present;
- no display process holding the GPU.

The first arm's block, with its placeholders, is in the design doc's §5.6.8. The literal stand-in line is:

```
# NEEDS GO. Stand-in values. Never run as written.
python -m main.launcher --restart-interval-hours 24 --pin-commit 9ebda1904da7a7782f22de2495b6e4c51368510b \
  $(cat designs/research_state/measurements/x35_breakers_stage_2026-10-17/argv_a1_s3501_standin.txt)
```

## 3. Pre-checks at the call (a failure is a HOLD; T55's `launch_x35.sh` will run them)

1. X26 and X34 have stopped, and the GPU lease is the Training session's to grant.
2. `X35_PIN_1` contains T54, the pool seed's step filter (F18).
3. No `models/rb_x35_*` directory for the arm exists.
4. The sha256 of P\* and of every archive node is unchanged since the call.
5. `checkargs --pin X35_PIN_1` exits 0.
6. The dry-run says "would launch", FORK of `rb_x26_s2601`, +20,000,000.
7. The reference population R is frozen and its file is written before the first arm launches (§5.6.4).
8. The disk has room for the whole batch, or finished arms are being thinned (F22).

## 4. The batch-1 queue (production first, then the toy calibration)

| # | run | seed | GPU-h (est.) |
|---|---|---|---|
| 0 | R0: offense-archetype best responder of P\* | 3531 | ≈ 1.1 |
| 1–6 | A1 s3501 · C0 s3501 · C0 s3502 · A1 s3502 · A1 s3503 · C0 s3503 | 3501–3503 | 6 × 4.4 ≈ 26 |
| 7–9 | A2 s3501–s3503 | 3501–3503 | ≈ 13 |
| 10–15 | toy A1 / C0 in the same ABBA order | 3601–3603 | ≈ 26 |
| 16–18 | toy A2 | 3601–3603 | ≈ 13 |
| — | R2 readers (1 per arm final, seed 1) and the population reads, in the windows between arms | 3532+ | ≈ 1.4 per arm |

The first A1-vs-C0 read lands after ≈ 27 GPU-h. A3 (≈ 18 GPU-h) follows in the same batch if the orchestrator wants
it before the toy calibration. That is a tactical call (`feedback_gpu_scheduling_is_mine`).

## 5. Findings at staging

These are F13–F24 in the design doc. The four that bind a launch:
- **F14.** A fork defaults to `--behaviour-check fatal`, so the argv types `warn`.
- **F16.** An exploiter fork needs `--no-self-play`.
- **F18.** The fork pool seed leaks X26's post-P\* snapshots; T54 fixes it.
- **F21.** The home72 config lives outside the repo.
