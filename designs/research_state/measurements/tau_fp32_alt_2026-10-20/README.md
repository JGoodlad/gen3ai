# FP32_ALT: X5's τ solved by a SPLIT residual in fp32 (2026-10-20, `gen3_tau_fp32_alt_v1`, CPU only)

## Verdict

**FP32_ALT meets T62's bar on X26's τ-class rows at fp32 cost, and is the DEFAULT** (`--tau-solve fp32_alt`, config
v160; T62's float64 stays selectable as `fp64`). On every violator, τ lands within 1.5e-6 of the exact root, and two
programs land on the SAME τ. The K9(b)-style learner-vs-rollout gap on the violators falls to T62's level or below.
It adds no float64 op. Its GPU cost is read by
[`../gpu_window_battery_2026-10-20/`](../gpu_window_battery_2026-10-20/) before X35's production GO.

**Why.** The 2026-10-19/20 GPU battery read T62's float64 solve at **+82–127 % of a training cycle**, against a 3 % bar:
- decision forward: B64 4.11 → 7.86 ms, B256 8.31 → 23.31 ms;
- learner: 41.4–42.0 s → 96.1–97.8 s.

Consumer cards run float64 at about 1/64 of the fp32 rate. FP32_ALT was the registered fallback.

## The formulation, and why it is well conditioned

`hypothesis_set.fixed_size_tau(..., solve="fp32_alt")` keeps T62's provable bracket and its 64 fixed bisection steps.
Only the direction test changes. With x = a + τ and H = {x > 0}:

    Σ σ(x)  >  k      ⟺      Σ_s h(x_s)  >  k − |H|,     h(x) = σ(x) if x ≤ 0,   −σ(−x) if x > 0

- Every term is ±σ(−|x|) ∈ [−½, ½], computed to a few ulp RELATIVE.
- k − |H| is an EXACT integer.
- The near-1 term σ(x) ≈ 1 − 1e-5 is never formed. Its 1 − π is what the naive fp32 sum loses below ulp(1).

**Proof.** Let R + D = Σ σ(−|x_s|), the absolute sum of the terms.
1. Any summation order errs by at most c·(R + D), with c ≈ (depth + 2)·u (u = 2^-24).
2. S' = Σ π(1 − π) ≥ ½ Σ min(π, 1 − π) = ½ (R + D).
3. log(R + D) is Lipschitz-1 in τ, since each term's derivative is at most the term.

So S(τ) − k exceeds the evaluation error whenever |τ − τ*| > −ln(1 − 2c) ≈ 2c. τ is fixed to about 2c ≈ 1e-6 absolute,
WHATEVER the slope. The naive fp32 sum's resolution is ulp(k) / S', which is 7e-3 – 2.8e-2 on these rows.

**Where it degrades.** Only when EVERY σ(−|x|) underflows in fp32, with the top candidate about 165 nats or more clear
of the next. τ then lands where the runner-up's log π floors at about −88.7, instead of its true −100 or below. These
weights are below e^−88 either way. Measured: τ error 8e-4 at a 160-nat lead, 2.2 nats at 180, 62 nats at 300.
Float64's own naive sum fails much earlier: off by 5e-4 at a 60-nat lead, and 13 nats at 100.

## 1. τ on the violators (`taus.py` → `taus.json`, `taus.log`)

The method is the A3 diagnosis's `locate_tau` / `tau_programs`, and T62's `repro.py`:
- a float64 decision forward captures every `fixed_size_tau` call's inputs;
- the species scores are cast to fp32, as both training programs hold them;
- each solve then runs eager, under 64 candidate orders, and inductor-compiled on CPU.

REF is the exact root of the fp32 scores (the split solve in float64; float64's naive sum agrees to ~1e-12 here).

| event @ checkpoint (k = 1) | slope | naive band | \|τ − REF\|: naive / **alt** / fp64 | order spread on violator: naive / **alt** / fp64 | compiled − eager on violator: naive / **alt** | predicted K9(b) gap (spread × ∂log π/∂τ): naive (rank / 33) / **alt** |
|---|---|---|---|---|---|---|
| u32180 @ 316000231 | 8.1e-6 | 1.5e-2 | 4.2e-3 / **1.4e-6** / 4.7e-7 | 1.2e-2 / **0** / 0 | 9.1e-3 / **0** | 7.6e-4 (1) / **0** |
| u32180 @ 317000041 | 1.0e-5 | 1.1e-2 | 5.1e-3 / **2.5e-7** / 2.5e-7 | 4.6e-3 / **0** / 0 | 0 / **0** | 1.1e-3 (1) / **0** |
| u32320 @ 317000041 | 1.7e-5 | 7.1e-3 | 3.0e-3 / **8.6e-7** / 8.6e-7 | 5.2e-3 / **0** / 0 | 2.8e-4 / **0** | 2.1e-4 (1) / **0** |
| u32320 @ 320000071 | 1.6e-5 | 7.5e-3 | 4.2e-3 / **1.5e-6** / 4.1e-7 | 9.5e-3 / **0** / 0 | 5.5e-3 / **0** | 2.2e-4 (1) / **0** |
| u35950 @ 353000151 | 4.2e-6 | 2.8e-2 | 1.7e-2 / **1.9e-7** / 1.9e-7 | 5.2e-3 / **0** / 0 | 0 / **0** | 3.3e-5 (1) / **0** |
| u35950 @ 354000424 | 6.5e-6 | 1.8e-2 | 2.5e-2 / **1.3e-6** / 6.3e-7 | 2.7e-2 / **0** / 0 | 1.7e-2 / **0** | 4.3e-3 (1) / **0** |

- **Over all 33 rows of every dump**, FP32_ALT stays within these limits:
  - |τ − REF| ≤ 1.8e-6;
  - order spread ≤ 1.9e-6;
  - compiled − eager ≤ 1.9e-6.

  That is ≤ 2 fp32 ulp of τ at |τ| ≈ 20. Float64's order spread and compiled − eager are exactly 0 (its τ is rounded once).
- **The move calls (calls 1–4)** were also ill-conditioned for the naive sum (band up to 4.1e-3, error up to 2.0e-3).
  FP32_ALT's absolute error there is ≤ 5.8e-7.
- **Synthetic saturated rows** (`tau_solve_test`): one or three candidates 12–150 nats clear of the rest, k = 1 / 3.
  FP32_ALT lands within 0.73 fp32 ulp of a 60-digit mpmath root, and within 2 ulp under 32 orders.

## 2. The network (`equivalence.py` → `equivalence_{A,B,C}.json` / `.log`)

The setup is T2's decision forward (`DecisionModule`, fp32, eager), under each solve, against a float64 forward (REF)
and an inductor-compiled CPU program of the same solve (the CPU analogue of T2's compiled rollout). The K9(b) analogue
is the LEARNER (`evaluate_actions`, train mode, eager, as the probe runs it) against the compiled decision forward, on
the taken action. Values are max |Δ log π| over the legal actions unless marked; greedy agrees on every row of every
block.

| block | FP32_ALT vs fp64 | program floor, compiled vs eager: naive / **alt** / fp64 | K9(b) analogue, learner vs compiled, taken action: naive / **alt** / fp64 |
|---|---|---|---|
| **A**: 915 fixture rows (`play_rows(24, seed=6201)`, T62's), P\* (X26 checkpoint_280000170); none saturated | 4.8e-5 (p99 2.1e-5) | 5.8e-5 / **8.4e-5** / 6.4e-5; p99 3.1e-5 / **2.6e-5** / 3.3e-5 | max 2.0e-5 / **1.5e-5** / 3.7e-5; p99 1.25e-5 / **1.0e-5** / 1.2e-5; rows > 5e-5: 0 / **0** / 0 |
| **B**: the 99 event rows (3 dumps × 33), P\*; 9 saturated | 2.3e-5 (saturated 1.8e-5) | 1.39e-3 / **8.4e-5** / 7.2e-5; saturated 1.39e-3 / **2.3e-5** / 1.5e-5 | max 2.6e-4 / **8.3e-5** / 4.0e-5; on the 3 violators 2.6e-4 / **≤ 8.6e-6** / ≤ 6.2e-6 |
| **C** u32180 @ 316000231 (the A3 stop's nearest weights) | 2.3e-5 | 5.8e-4 / **5.3e-5** / 7.3e-5 | violator **5.9e-4 / 2.4e-6 / 2.1e-5**; max 5.9e-4 / **2.6e-5** / 3.8e-5 |
| **C** u32320 @ 317000041 | 5.1e-5 | 5.3e-4 / **5.4e-4** / 5.8e-4 (a NON-τ row: saturated 9.5e-6 / **9.5e-6** / 7.2e-6) | violator 1.9e-6 / **6.7e-6** / 1.2e-5; max 4.1e-4 / **4.1e-4** / 4.7e-4 (the same non-τ row in every arm) |
| **C** u35950 @ 353000151 | 4.4e-5 | 1.8e-4 / **2.0e-4** / 1.9e-4 | violator 1.1e-5 / **1.4e-5** / 3.6e-6; max 6.7e-5 / **6.0e-5** / 1.1e-4 |

Error against REF:
- B: naive 2.2e-3, **alt 7.3e-5**, fp64 7.9e-5.
- C: u32180 2.7e-4 / **7.8e-5** / 8.2e-5; u32320 1.2e-4 / **6.1e-5** / 5.2e-5; u35950 1.16e-3 / **1.6e-4** / 1.5e-4.
- A: the same 3.0e-2 row in every arm, p99 2.9e-5 / **2.8e-5** / 2.8e-5. That is a non-τ fp32-sensitive fixture row, as in T62's own read.

**Reading.**
- **The τ class is gone under FP32_ALT.** At the A3 stop's nearest weights, the violator's learner-vs-rollout gap is
  5.9e-4 under the naive sum, 2.4e-6 under FP32_ALT and 2.1e-5 under fp64. On every violator FP32_ALT is ≤ 1.4e-5.
- **Away from the τ class, FP32_ALT sits at the program floor.**
  - Its taken-action K9(b) analogue on the 915 fixture rows is the lowest of the three solves: max 1.5e-5, p99 1.0e-5,
    no row above 5e-5. The bar is 1e-4.
  - Its all-actions program floor max is 8.4e-5 on A and B, against fp64's 6.4e-5 / 7.2e-5, while its p99 is equal or
    lower. FP32_ALT's τ sits within ~1 ulp of fp64's, so two programs can land on adjacent floats. That adds ulp-scale
    wobbles of up to about 2e-5 in log π, the class the naive sum already had on every well-conditioned row.
- **The 4e-4 rows at u32320's neighbour checkpoint are NOT τ.** They are equal in all three arms, and the saturated rows
  there sit at 1e-5. Some other fp32-sensitive mechanism acts on those scan rows at those weights. FINDING F4.

## 3. CPU cost (`perf.py` → `perf.json`, `perf.log`; ONE torch thread, arms interleaved, medians)

| | naive fp32 | **FP32_ALT** | fp64 (T62) |
|---|---|---|---|
| `fixed_size_tau` alone, species [512, 400], eager | 30.7 ms | **45.3 ms (×1.47)** | 109 ms (×3.55) |
| the same, inductor CPU (scalar codegen, `CPU_INDUCTOR_OPTIONS`) | 67.4 ms | **72.7 ms (×1.08)** | 156 ms (×2.32) |
| per-slot moves [512, 6, 371], eager | 249 ms | **376 ms (×1.51)** | 1,053 ms (×4.23) |
| the same, inductor CPU | 320 ms | **492 ms (×1.54)** | 574 ms (×1.80) |
| T2 decision forward, eager, B 1 / 16 / 64 / 256 | 102 / 164 / 370 / 1,617 ms | **+3.8 / +6.7 / +5.3 / +10.1 %** | −1.1 / +6.0 / +2.4 / +41 % |
| T2 decision forward, inductor CPU, B 64 | NOT READ: 6,732 ms / 385 ms / 984 ms after compiles of 44 / 378 / 32 s — the naive arm's compiled call is 18× its own eager call, so a recompile (or a cache artefact) sits inside the timing, not a cost (FINDING F3) | | |

- **The CPU cost is ~fp32.** FP32_ALT's test does about 2× the naive test's elementwise work per element: abs, a
  sigmoid, a select, and a fused integer count. On the eager decision forward that is +4 to +10 % at one thread.
- **The CPU does not decide the gate.** CPU float64 runs at full rate, so this table understates fp64's GPU cost.
- **Contention.** The box ran other agents' jobs (load 10–16), so treat ±5 % as noise. T62's own one-thread read put
  fp64 at +6.4 % at B 256; this run reads +41 %.
- **The GPU read** is `../gpu_window_battery_2026-10-20/`: FP32_ALT vs the naive sum, bar 3 % of a cycle.

## 4. The goldens moved by the default switch (bisected)

| golden | old | new | proof of cause |
|---|---|---|---|
| K9 learner golden, post / fp32 opt-out (+cu126 = +cu128) | `bd79d729…` / `d614ec22…` | `584de31c…` / `318c114b…` | every hypothesis builder forced to `fp64`: the T62 record reproduces bit for bit (21 / 21 golden + fixed-mass + sentinel tests) |
| played-bits sentinel `PLAYED_DIGEST` (`PACKED_DIGEST` unchanged) | `6b2517a4…` | `41a35939…` | the same forced-`fp64` run reproduces `6b2517a4…` |
| reveal file `OFF_BITS` (`OFF_OUTCOME_ALL` `26877212…` unchanged: every result, end turn, team and action the same) | `a5cd83ef…` | `7e97c03f…` | `reveal_off_digest.py fp64` (the file's own OFF/OFF play, the solve forced) reproduces `a5cd83ef…` exactly (`reveal_off_digest.log`) |

## Files

- **Scripts:** `taus.py`, `equivalence.py` (`--blocks`, `--events`, `--compiled`), `perf.py`, `reveal_off_digest.py`.
  `equivalence_B.json` was written by the revision before the learner path's tail statistics (`p999`, the over-bar
  counts, the per-row array) were added; nothing else differs.
- **Inputs (read only):**
  - `~/.cache/gen3ai/x26_a3_stop/` (u32180, u32320 and checkpoints 316000231 / 317000041 / 320000071);
  - `~/.cache/gen3ai/x26_a6_u35950/` (u35950, checkpoints 353000151 / 354000424);
  - the policy template `~/.cache/gen3ai/x26_a6_rediag/u35950/in/checkpoint_316000231_steps.zip`;
  - P\* `~/.cache/gen3ai/x35_refpop/pstar/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`.
- **Run as:** `CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh … timeout 45m`, with
  `GEN3AI_BLOCKING` (this unit blocks X35's production arms). Peak memory was ≤ 7.1 GB. No GPU was touched.

## FINDINGS (standing rule 7)

- **F1. The cross-program residue is ulp-scale, not zero.** FP32_ALT's τ sits within about 1 fp32 ulp of the float64
  root, so two programs can land on adjacent floats where fp64's land on the same one. That adds wobbles of up to
  about 2e-5 to log π. These are the class of the naive sum on every well-conditioned row, which X26 lived with for
  380M steps at a K9(b) p99 of about 2.2e-5. On the taken action this read finds no excess. The all-actions program
  floor max is 8.4e-5, against fp64's 6.4e-5 / 7.2e-5 on A / B. A live run's K9(b) record is the confirmation.
- **F2. The GPU cost is unmeasured here.** On the CPU, FP32_ALT's test does about 2× the naive test's elementwise work.
  On a GPU, inductor fuses each bisection step into one kernel either way, so the cost should stay near fp32's.
  **UNVERIFIED** until `../gpu_window_battery_2026-10-20/` runs.
- **F3. The compiled CPU forward read is anomalous and not read** (§3). The naive arm's compiled call was 18× its
  eager call. Cause not chased: the CPU compile is not the gate's measure.
- **F4. A non-τ tail at u32320's neighbour checkpoint.** The scan rows there show learner-vs-compiled gaps of about
  4e-4 in ALL three arms (fp64 included), with the saturated rows at 1e-5. Some other fp32-sensitive mechanism is at
  work, and FP32_ALT neither causes nor fixes it. Not investigated: out of scope.
- **F5. FP32_ALT degrades past a ~165-nat lead** (every σ(−|x|) underflows). This is far beyond today's rows, which
  lead by about 23 nats, and float64's naive sum fails first, from about 60 nats. A log-space residual would remove
  the limit at a logsumexp per step.
- **F6. The A3 / A6 diagnosis scripts** (`x26_a3_stop_316M_2026-10-18/locate_tau.py`, `tau_programs.py`) and T62's
  `repro.py` / `equivalence.py` wrap `fixed_size_tau` with a fixed signature, and patch `TAU_SOLVE_DTYPE`. At this
  commit they fail LOUD: a TypeError on the new `solve=` keyword, or a patch of a constant that no longer exists.
  They are pinned records for X26's tree, which is unaffected. This directory's `taus.py` is their port.
