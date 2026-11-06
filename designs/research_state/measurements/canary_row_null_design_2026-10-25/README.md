# The canary's per-row dither-null clip margin: DESIGNED 2026-10-25; BINDING at new pins since 2026-10-28

**Status (2026-11-05).** The design is built, and k is set and replicated:

- **The null** (rule 1, verbatim: 64 unique rows, K = 16 name-keyed seeds, relative 2^-9, fp32 'highest') runs at every
  canary (`gen3_canary_row_null_shadow_v1`, `src/agents/model/canary_row_null.py`). Measured cost: CPU 3.3-3.5 s on the
  K9 golden learner; CUDA 2.2-2.4 s at K = 16 (each census's `cost_cuda.json`; an upper bound, the noise is drawn on the CPU); capped at 5 s.
- **The bar** binds the compiled arm at new pins since `gen3_canary_row_bar_v1` (stage 1, `canary_row_bar.py`):
  b_i = min(k x max(null_i, 0.02), 0.72), **k = 6.0**. The rule that set it, and the rulings on rules 2-5, are
  `designs/training/canary_recalibration_2026-10-28.md` §1. As built it differs from rules 3 and 4 below: relocation
  stays ONE scalar (the cap), and the yardsticks keep the scalar rule. The rules as built are
  `designs/training/compile_flags.md` "The per-row clip margin, STAGE 1".
- **k's rule (ruled 10-28):** k = 4 (`R1_BAR_K`) x the healthy maximum of shift / max(16-seed null, 0.02) over the
  canary's own two bf16 arms at B = 2,048 on CUDA (compiled and eager pooled), rounded up to 2 significant digits. On
  the W0 census that maximum is 1.479 (eager, X26 380M, row 63), so k = 6.0. `canary_row_bar_test` re-derives it from
  the committed basis (`src/agents/model/testdata/canary_stage1_basis.json`, W0's reads).
- **The replicate (2026-11-05) re-derives k = 6.0** at both NVIDIA drivers. Section "The census and its replicate"
  below.

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

## Open questions at design time, and how each closed

1. **k.** CLOSED: 6.0, from the W0 CUDA census (10-28), replicated (11-05). The CUDA compiled point that suggested
   10.7 was cross-weight: on the same weights and device the compiled arm reads at most 1.17 x its row's null.
2. **The null's device.** CLOSED: on CUDA ('highest', cuBLAS) the fp32 null equals the CPU null for the same dithers to
   at most 1.4e-4 log-prob on every row: 11 checkpoints read on both, CPU in `../x26c_canary_margin_2026-10-25/results`,
   CUDA in the replicate, at either driver. That is under 1 % of the 0.02 floor.
3. **K, and the jump rows.** CLOSED for stage 1: K = 16 (`canary_recalibration_2026-10-28.md` §1; K = 8 would set k at
   13). A jump row still rests on the seeds that cross it.
4. **Relocation and the gradient rule's bars.** CLOSED by construction: relocation stays one scalar, and every bar is
   capped at it (stage 1: 0.72).
5. **Teeth on tail rows.** Stage 1's cap keeps every bar at or under M. Stage 2 (cap 1.2) is
   `designs/ops/TECH_DEBT_BACKLOG.md`'s row; its 10-29 read was INCONCLUSIVE.
6. **K6 lifecycle.** **UNVERIFIED** here: this unit read no production record.

## The census and its replicate (2026-11-05; `census_replicate.py`, `results/census_replicate_2026-11-05.txt`)

The cheaper path (k from an era of production canaries) could not run: no production pin carried the shadow
(`canary_recalibration_2026-10-28.md` FINDING 1). k came from `gpu_census.sh` instead, which ran twice on the same 20
X26-lineage checkpoints. Both runs used the same script and the same env (torch 2.8.0+cu126, triton 3.4.0).

- **W0** (`~/.claude/jobs/x35/w0_2026-10-28/census/`): all 20 on 10-28, under NVIDIA driver 595.91.07. This is the
  basis k was ruled on.
- **The replicate** (`~/gen3ai_archive/canary_census_2026-10-25/`, checkout `742decb2e`): 14 checkpoints (100-460M) on
  10-25 under 595.91.07. The last 6 (470M, 480-482M, the u1000 neighbours 478M / 479M) ran on 11-05 under 595.99.02
  (installed 11-03 with `linux-modules-nvidia-595-open` for kernel 7.0.0-42). Its `derive_k.py` output
  (`derive_k_2026-11-05.txt`, sha256 `ec2179be…`) re-runs byte-identically on CPU.

**The driver changes nothing that sets k.** The two runs overlap on every checkpoint, so the test is direct, on the
same weights:

- Every eager quantity is **bit-identical** to W0's at both drivers: the 32-seed fp32 null, per seed and per row;
  eager bf16 at B = 64 and B = 2,048; and the dithered eager arm.
- **The compiled arm is not reproducible across processes, at either driver.** Each read compiles afresh into a
  private cache. Against W0's draw, a row's compiled shift moves by:
  - same driver: median 0, p99 0.050, max 0.138 log-prob; 25 % of rows bit-equal;
  - across drivers: median 0, p99 0.060, max 0.102; 17 % bit-equal.

  The cross-driver spread sits inside the same-driver one. One checkpoint, 481M, moves its compiled maximum by +0.18
  (1.147 against W0's 0.969). That is more than any same-driver checkpoint (at most ±0.11). One point cannot separate
  the driver from the draw, and the value is still under the binding 1.479.

**k under the ruled rule** (16-seed null, floor 0.02; the maximum ratio, then k = round_up_2sig(4 x max)):

| census | compiled (Cbf) | eager (Ebf), B = 2,048 | the rule (both pooled) |
|---|---|---|---|
| W0, 20 checkpoints (the basis) | 1.170 (X26 100M, row 4) -> 4.7 | 1.479 (X26 380M, row 63) -> 6.0 | **6.0** |
| replicate, 20 | 1.157 (X26 100M, row 4) -> 4.7 | 1.479 (bit-identical) -> 6.0 | **6.0** |
| replicate, old driver, 14 | 1.157 -> 4.7 | 1.479 -> 6.0 | 6.0 |
| replicate, new driver, 6 (a subset) | 1.147 (X26c 481M, row 61) -> 4.6 | 1.408 (X26c 480M, row 29; W0's too) -> 5.7 | 5.7 |
| both compile draws pooled | 1.170 -> 4.7 | 1.479 -> 6.0 | **6.0** |

- Stage 1 (k 6.0, cap 0.72) refuses **0 of 20** of the replicate's compiled reads. The nearest reads 0.48 of its bar
  (470M).
- Eager at B = 64, which the canary never runs, reads 1.734 (X26c 480M, row 29), which gives 7.0. It is not in the
  rule.
- **`derive_k.py`'s headline is not the ruled statistic.** It uses the 32-seed null and floors 0.01-0.1, and it lists
  each arm alone. For example: compiled 1.027 -> "k = 4.11", eager at 2,048 "5.63", eager at 64 "6.93". The canary's
  null has 16 seeds, and the maximum over 16 is at most the maximum over 32. So a k from the 32-seed column would sit
  below the healthy tail of the null the canary actually computes.

## Files and how to run

- `derive_k.py <dir> …` (CPU, seconds) prints the calibration table above from any directory of `margin_read.py` reads.
  `python derive_k.py ../x26c_canary_margin_2026-10-25/results` reproduces `results/calibration_cpu.json`.
- `null_cost.py --ckpt <zip> --out cost.json [--device cpu|cuda] [--k 16] [--vf-coef 0.5]` measures the cost.
  `--vf-coef` is typed through because a re-read of an X26-lineage checkpoint is otherwise refused (finding F7 of the
  x26c README; the value is the run's `model_config.json` 0.5, and it weights the critic only). F7's root cause was a
  checkpoint zip the run's retention had REMOVED; since 2026-10-25 that is refused as "checkpoint file not found"
  (`checkpoint_state.require_checkpoint`), so `--vf-coef` is no longer needed for a zip that exists.
- `census_read.py --run-dir <models/run> --at 330M --out D [--device cuda] [--dither 32]` reads one checkpoint, every
  fixture row as a row of interest, with `--compiled`. At this checkout it reproduces the pin's 478M eager per-row read
  bit-identically (max difference 0.0, CPU smoke).
- **`gpu_census.sh <clean checkout> <out>` is the GPU-window script (lease-gated; it ran twice, above).** It reads 20
  X26-lineage checkpoints on CUDA: X26 at 100-380M, X26c at 390-482M, and the archived u1000 neighbours 478M / 479M,
  which are copied into a scratch run dir. Then it measures the null's cost on the card. About 5 min per checkpoint.
  It is resumable, but it rewrites `checkout_head.txt` on every invocation, so a resumed census records only its LAST
  checkout. Then run `derive_k.py <out>` on CPU.
- `census_replicate.py <replicate dir> <W0 census dir>` (CPU, seconds) prints the driver test and the ruled k above
  (`results/census_replicate_2026-11-05.txt`).

## Decision record

| date | decision | why |
|---|---|---|
| 2026-10-25 | Design only; the near-margin dump built instead | open questions 1-4 move a bar or the gradient rule's fixture on CPU data alone, and the only CUDA points are cross-weight |
| 2026-10-25 | The null built as a recorded-only shadow in every canary (orchestrator's go) | it closes open questions 1-3 from production CUDA canaries without a GPU window, and judges nothing |
| 2026-10-28 | k = 6.0 RULED, and stage 1 built (`gen3_canary_row_bar_v1`; `canary_recalibration_2026-10-28.md`) | no production pin carried the shadow, so the W0 CUDA census is the basis |
| 2026-11-05 | The replicate census: k = 6.0 re-derives; the driver change (595.91.07 -> 595.99.02) leaves every eager quantity bit-identical; no code change | the ruled rule on the replicate, and on both compile draws pooled, gives 6.0 |
