# The canary's per-row dither-null clip margin: DESIGNED, NOT BUILT (2026-10-25)

**Status.** A design with its calibration evidence and the scripts that settle it. Not built: four judgement calls below
cannot be settled from the CPU data (open questions 1-4), and the brief's rule was "build only if it is clean". Built in
the same unit instead: the near-margin weight dump (`gen3_canary_near_margin_dump_v1`, `compile_canary.py`), so the next
near-miss leaves its weights behind. Source of the problem: `../x26c_canary_margin_2026-10-25/README.md` (fa682e92), and
the TECH-DEBT canary row (`designs/ops/TECH_DEBT_BACKLOG.md`, Next-10 #1).

## The problem in one paragraph

R1's clip-margin rule (`compile_regions._clip_margin_rule`) holds every arm to ONE global margin, M = `R1_BAR_K` 4 ×
`R1_LOGPROB_SHIFT_MAX` 0.18 = 0.72 log-prob. 0.18 came from a CPU eager census of a toy run at 1-62M. On X26-lineage
weights the fp32 function ALONE moves some fixture row by 0.80-1.23 under a half-bf16-ulp weight dither from 330M on,
and the compiled arm read 0.712 at the X26 continuation's u1000 (1 % headroom). The compiled arm has no M′ re-run, so a
healthy tail trips UNCONFIRMED and then FATAL. And M is too LOOSE where the function is quiet: on 69 % of fixture
row-reads a per-row bar built from the row's own null sits below 0.72 (table below), so a miscompile that moves a quiet
row by 0.5 passes today.

## The rule as designed

1. **The null, at the canary's own check time and on its own device.** Before the rung relocates its rows, run the fp32
   REFERENCE forward (`fp32_scope`, no grad, train mode, matmul precision 'highest') on the 64 UNIQUE fixture rows (B = 64,
   not the tiled 2,048: E32 at 2,048 equals E32 at 64 to 2.9e-5). Then run K seeded weight dithers of it. Each dither is a
   relative 2^-9 multiplicative perturbation, name-keyed as in `diag_bf16_canary.dithered`. The noise is drawn from a
   generator the canary owns, ON the device, and applied through `torch.func.functional_call`, so no live parameter is
   ever mutated. The row's null is `null_i` = max over the K seeds of |log π_s,i − log π_i|.
2. **The per-row bar.** b_i = k × max(null_i, f). The compiled arm FAILS row i iff its |Δ log π| from the reference on
   that row exceeds b_i, taking the max over the row's tiles at B = 2,048. A row with no shift and no null passes at the
   floor k × f.
3. **Relocation per row.** A row within its own b_i of its PPO switching boundary moves 2 b_i inside the active side.
   This keeps rule 8's categorical statement: no arm inside its bars can switch a row. Today `relocate_clip_boundary`
   takes one scalar margin, so it would take a vector.
4. **The yardsticks** (eager bf16, the fp32 reference) get the same per-row bar. One out of b_i is the recorded event,
   and the rung re-runs once with that row's bar at `YARDSTICK_HEADROOM` × the yardstick's shift, as M′ does today.
5. **Shadow for one era.** The old global rule is computed on the same arms and recorded beside the verdict
   (`watch.*.clip_margin.shadow_global` {margin, verdict, fraction}), never judged, so the two can be compared run by run.
6. **The record** carries `null_per_row`, `bar_per_row`, each arm's per-row shift and the K seeds' worst row, so a growth
   curve of the conditioning is recoverable from every run.

## The calibration so far (CPU only; `derive_k.py`, `results/calibration_cpu.json`)

The healthy ratio is an arm's per-row |Δ log π| from the fp32 reference divided by max(that row's 16-seed fp32 null, f),
over the x26c margin read's 11 checkpoints (100M-479M, B = 64) and its CPU Inductor compiled read (478M, B = 2,048):

| arm | row-reads | old rule (M 0.72) refuses | ratio p50 / p90 / p99 / max (f = 0.02) | worst | k = 4 × max | per-row bar median / max | bar < M on |
|---|---|---|---|---|---|---|---|
| eager bf16, B = 64 | 704 | 0 | 0.137 / 0.466 / 0.845 / **1.093** | 330M row 19 (0.123 / 0.112) | 4.37 | 0.425 / 5.39 | 69 % |
| CPU Inductor compiled bf16, B = 2,048 | 64 | 0 | 0.121 / 0.438 / 1.026 / **1.071** | 478M row 21 (0.279 / 0.260) | 4.28 | 0.424 / 3.55 | 64 % |
| eager bf16, B = 2,048 | 64 | 0 | 0.146 / 0.401 / 0.738 / 0.985 | 478M row 15 | 3.94 | 0.390 / 3.26 | 66 % |

- **On CPU, a healthy bf16 arm, compiled or eager, never moves a row by more than about 1.1 × its own fp32 null.** That
  is a much tighter, row-shaped family than one global M.
- **The floor is not sensitive.** On 103 of the 704 row-reads the null is under 0.01. bf16 moves those rows by at most
  0.0063, so every floor from 0.01 to 0.1 gives the same maximum.
- **The CUDA points do not fit the CPU family.** There are two, each cross-device AND cross-weight:
  - The x26c u1000 compiled 0.712 on row 50 (478.31M) against row 50's CPU null 0.267 at 478.0M is a ratio of
    **2.67**.
  - The X34 u1700 eager 0.856 against the largest CPU null of 0.54 (267M) is a ratio of **≥ 1.6**.

  Under R1's convention (bar = `R1_BAR_K` × the healthy maximum), 2.67 alone would set **k ≈ 10.7**, not 4.4.
- **The null's max is heavy-tailed in K.** Row 50's 16 seeds at 478.0M read a median of 0.055, but one seed reads 0.267
  and the next highest 0.145. At 479M the row is a JUMP row (a discrete op flips under the dither): 5 of 16 seeds read
  0.31-0.47. So the compiled arm's 0.712 at 478.31M sits between a smooth-row null and a jump-row null, three updates
  apart. Which one it was is UNVERIFIED (the weights were not dumped; the near-margin dump now keeps them).

## Cost

- **CPU** (`null_cost.py`, `results/cost_cpu_482M.json`, 482M; contention-bound: load average 12.5 on the agent cores at
  nice 19 while rb_x26c trains):
  - one fp32 forward at B = 64 takes 7.1-9.3 s, and the K = 16 null 126 s;
  - one forward at B = 2,048 takes 23-30 s (3.3× the B = 64 one);
  - each dither's apply + restore takes 0.48 s.

  The same forwards ran about 20× faster in the x26c read's quieter session (35 forwards in 13 s), so the absolute
  numbers here are an upper bound, not a cost.
- **GPU (estimate, UNVERIFIED).** The canary's two B = 2,048 forward+backward arms cost 0.65-0.71 s (K6). A B = 64 fp32
  forward is launch-bound, roughly 0.03-0.1 s, so K = 16 comes to about 0.5-1.6 s per canary with on-device noise. That
  is under 0.05 % of the ~1 h between canaries. The transient is the forward's activations at B = 64, plus one
  dithered-parameter dict of ~40 MB; `update_fit`'s dry canary would measure it. `gpu_census.sh` measures the seconds,
  but with `diag_bf16_canary.dithered`'s CPU-side noise, so it reads high.

## Open questions (why this was not built)

1. **k.** CPU says 4.4, and the one CUDA compiled point says at least 10.7 if the CUDA null on row 50 matches the CPU
   null. The healthy CUDA compiled / null ratio has never been measured on the same weights and the same device. **This
   needs the CUDA census.**
2. **The null's device.** Is a per-row null computed on CUDA ('highest', cuBLAS) the same per row as the CPU null? The
   bar would be computed on CUDA in production. The census reads both sides.
3. **K, and the jump rows.** The max over K seeds catches a discontinuity only if one of K seeds crosses it. Row 50's
   null at 478.0M rests on one seed of 16. K = 16 may be too few; a seed quantile would be steadier but has no teeth
   against a jump. The census records every row's per-seed null at K = 32 (`derive_k.py` prints the K-sensitivity).
4. **Relocation and the gradient rule's bars.** Per-row relocation moves the tail rows up to 2 × b_i ≈ 10 log-prob (k
   4.4) or ~25 (k 10.7) inside the active side. The bf16 DRIFT / ENVELOPE bars were derived (window D) on rows relocated
   at the global M. Keeping relocation at M while judging the margin per row re-opens the 0/1 clip lottery on any row
   whose shift lies between M and b_i (rule 8). Either way the gradient rule's inputs change, and its bars would need a
   re-read. That is a design choice beyond the margin.
5. **Teeth on tail rows (the orchestrator's call).** On a tail row the bar rises to ~5 (k 4.4) or ~13 (k 10.7)
   log-prob. A miscompile confined to that row passes the margin rule. The gradient rule still sees it, and on the
   other 64-69 % of rows the bar tightens. Is a looser bar on 1-3 % of rows an acceptable trade?
6. **K6 lifecycle (UNVERIFIED).** A device generator and a dithered-parameter dict created at canary time are plain
   tensors, not parameters, modules or buffers. Nobody has checked that the K6 acquisition guard accepts them after the
   freeze.

## The cheaper path to k (recommended next step, needs the orchestrator's go)

Ship the null as a **recorded-only shadow** inside the canary: compute `null_per_row` and each arm's per-row
shift / null ratio at every canary, judge nothing, and keep the old rule binding. Every production canary then adds one
row of the CUDA census on its own weights and its own kernels, for about 1 s an hour. After one era, k, K and the floor
come from real CUDA reads, and open questions 1-3 close without a GPU window. This adds K forwards to every production
canary (cost and memory: the GPU estimate above), so it is not built here unasked.

## Files and how to run

- `derive_k.py <dir> …` (CPU, seconds) prints the calibration table above from any directory of `margin_read.py` reads.
  `python derive_k.py ../x26c_canary_margin_2026-10-25/results` reproduces `results/calibration_cpu.json`.
- `null_cost.py --ckpt <zip> --out cost.json [--device cpu|cuda] [--k 16] [--vf-coef 0.5]` measures the cost.
  `--vf-coef` is typed through because a re-read of an X26-lineage checkpoint is otherwise refused (finding F7 of the
  x26c README; the value is the run's `model_config.json` 0.5, and it weights the critic only).
- `census_read.py --run-dir <models/run> --at 330M --out D [--device cuda] [--dither 32]` reads one checkpoint, every
  fixture row as a row of interest, with `--compiled`. At this checkout it reproduces the pin's 478M eager per-row read
  bit-identically (max difference 0.0, CPU smoke).
- **`gpu_census.sh <clean checkout> <out>` is the GPU-window script: WRITTEN, NOT RUN, lease-gated.** It reads 20
  X26-lineage checkpoints on CUDA: X26 at 100-380M, X26c at 390-482M, and the archived u1000 neighbours 478M / 479M,
  which are copied into a scratch run dir. Then it measures the null's cost on the card. Estimate 45-75 min.
  Then run `derive_k.py <out>` on CPU.

## Decision record

| date | decision | why |
|---|---|---|
| 2026-10-25 | Design only; the near-margin dump built instead | open questions 1-4 move a bar or the gradient rule's fixture on CPU data alone, and the only CUDA points are cross-weight |
