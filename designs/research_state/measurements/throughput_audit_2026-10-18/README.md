# The throughput audit: why the Gen 9 VGC bots train ~75× more env steps per second than we do, and which levers matter (2026-10-18)

**The question (owner, 2026-10-18):** "were there any obvious gaps from what we're doing versus what others are doing?"
The survey [`../../literature/showdown_bots_landscape_2026-10-18.md`](../../literature/showdown_bots_landscape_2026-10-18.md)
quotes mikumiku37 (the first #1 Gen 9 VGC bot) at about 100,000 env steps/s on one RTX 5090. X26, our production end
state, runs at about 1,350 on an RTX 3080 Ti. This audit explains the gap and ranks what we could change.

**What it is:** analysis plus short CPU measurements. No GPU (X26 held it), nothing launched, nothing written under
`models/` or `data/`. All CPU work ran under rule 5b (`nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh`,
`CUDA_VISIBLE_DEVICES=""`), one job at a time, beside X26 and the agents' maturity and team-strength queues (load
average 11–13 on 16 threads). The GPU measurements are WRITTEN, not run (§5).

"Step" below means one TRAINEE decision: one row of the rollout buffer, the unit of `time/fps`.

## 1. Bottom line

1. **The ~75× is not one gap. On the same card it is two stacked gaps of different kinds.**
   - **The card:** an RTX 5090 runs our cycle about **1.75× faster** than our 3080 Ti (the cloud bench's Amdahl rule,
     re-applied to X26's current cycle).
   - **The update (the PPO learning step), ≈ 28× per sample on the same card.** Most of it is that we take **10 passes
     (epochs) over every rollout**, where their published numbers only fit about 1–2 passes. The rest is that our
     network uses the GPU poorly: about **6–8 % of the card's peak** on the matrix arithmetic we count, because it is
     thousands of small memory-bound operations (the damage op, the edges, the beliefs). **It is NOT model size:** their
     network is about 3.3× our parameters, and its forward pass is probably 1–3× our arithmetic.
   - **Collection and overheads, ≈ 75× per sample on the same card.** About half is inference spread thin: every host
     step runs about 21 separate small forward passes (the trainee plus up to 20 frozen opponent snapshots, about 11
     rows each), so one inference row costs **88 µs of GPU wait, about 6× the ≈ 14 µs** the same forward costs inside
     the learner (42.7 µs per row per pass for forward + backward; the forward is about a third). They batch 8,192
     battles at once. The other half is our env core (7.6 % of the
     cycle), Python glue (3.5 %), and in-loop eval and startup (6.9 %), all serial with the GPU.
2. **Their numbers are not one unit.** 330M games in 48.3 h is 1,898 games/s, so 100,000 steps/s implies 52.7 "steps"
   per game, but a Bo1 VGC game lasts about 6–12 turns. The real per-decision gap is therefore somewhere between about
   **15× and 75×** (FINDING F1). Jaxcalibur, a singles bot (one H100, 100M games in a week, 165 games/s), is only about
   **1.5–6× our rate** in decisions.
3. **The env core's protocol pipeline is not the bottleneck.** It is 7.6 % of X26's cycle. Making it infinitely fast
   would buy at most ×1.08. Its per-decision cost is about 90 % protocol-text round trip, allocation and encoding, and
   only about 2–3 % battle simulation (§4.2).
4. **The top three levers, by expected gain × risk (§6):**
   1. **Fewer PPO epochs**, at the same number of optimizer steps per rollout: ×1.39 (5 epochs) to ×1.81 (2 epochs) on
      today's cycle. It carries a real sample-efficiency risk, and the cheapest A/B is about 10 GPU-h (X44).
   2. **Cut the opponent inference fan-out**, either by capping the snapshots live in one rollout or by a batched
      multi-snapshot forward: est. ×1.08–1.20, with low risk. Tomorrow's GPU cells size it first (X45).
   3. **Overlap collection with the update** (X14, already in the backlog): est. ×1.15–1.3 today, and more once the
      epochs fall, because collection then dominates. It is the biggest build and carries a policy-lag risk.
   - **Applied together on the 3080 Ti:** 73 s → about 24–30 s per cycle, i.e. ×2.4–3. With the 5090's ×1.75 on top,
     ≈ 5,500–7,000 steps/s.
   - **What is still left after that:** a ~15× residual, which is the network's per-row GPU efficiency (many small
     memory-bound ops). Closing it is an ARCHITECTURE question, and the owner's 2026-10-16 framing rules out a slimmed
     production (`../../../endstate/design_compute_efficiency.md` §6).

## 2. Data

- **X26** (`rb_x26_s2601`, pin `9c033489`, RTX 3080 Ti, bf16 trunk, 256 envs, a 20-snapshot pool, self-play 0.90).
  Medians over the last 30 rollouts at about 312M steps, read from its TensorBoard events on 2026-10-18 at
  ~20:10 PDT (read-only):
  - **the cycle:** `train/train_ms` 41,962; `rollout/collect_ms` 26,065; `time/fps` 1,346; `rust_env/host_steps` 416;
  - **per host step:** `gpu_wait` 40.20 ms, `core` 13.33 ms, `submit` 1.65, `flush` 2.37, `opp_draw` 0.45, `draw` 0.17,
    `write` 0.72, `post` 0.74; `fill` 0.27 s per rollout;
  - **per rollout:** 236.4 trainee rows per host step, 91,040 opponent (p2) policy rows, 2,228 games ended;
    `rollout/ep_len_mean` 44.8.
  - The cloud bench's own calibration (`gpu_bench_suite.LIVE_CYCLE`, at 258.6M: 42.24 / 26.97 / 5.6 s, 1,314 fps)
    agrees to 2 %.
- **Our network's cost:** `../compute_efficiency_2026-10-16/` §1.2. Counted matmul-family FLOPs are 94.8 MFLOP per row
  forward and 244.6 per row forward + backward (the production micro-step). Parameters at HEAD: **2,609,183**
  (`param_count.py`), of which embeddings are only 32,352.
- **The update's per-epoch split:** the frozen GPU suite's 1- and 3-epoch medians
  (`../gpu_bench_2026-10-18/suites/rtx3090_hu.json`): E1 5.05 s, marginal epoch 3.89 s, i.e. a fixed 1.16 s per update.
  It is scaled to X26's measured E10 (×1.048).
- **Their side:** the survey's self-reported figures (mikumiku37: 8.7M params, 8,192 parallel battles, 32 steps per
  update = 262,144 samples, ~100k steps/s, PPO gradient ~70 % of wall, Rust engine ~7k battles/s per core;
  Wolfe_Alt6767: 7.5M params, 8 layers × width 256, 38k battles/s per thread; Jaxcalibur: 8.5M MoE, 100M games/week on
  one H100). Every derived number on their side is an ESTIMATE.

`gap_decomposition.py` holds every input with its provenance and prints every number in §3 and §6
(`result/gap_decomposition.txt`).

## 3. Normalising the comparison

| quantity | ours (X26) | mikumiku37 | Wolfe_Alt6767 | Jaxcalibur |
|---|---|---|---|---|
| format | Gen 3 OU singles | Gen 9 VGC doubles, Bo1 | same | Gen 9 random battles, singles |
| parameters | 2.61M (embeddings 32k) | 8.7M | 7.5M (8 layers × 256) | 8.5M (mixture of experts: fewer active) |
| forward MFLOP per decision | 94.8 counted (64 tokens × 3 rounds at width 128, plus the damage op, edges and beliefs) | est. 100–300 if shaped like Wolfe's (8–24 tokens × 12.6 MFLOP per token) | same estimate | unknown |
| decisions per env step | 1 | 2 slots, one forward (slot 2 conditioned on slot 1) | 2 | 1 |
| trainee steps per game | 44.1 | stated figures imply 52.7; real VGC games ~6–12 turns (F1) | — | unknown (randbats games ~15–50 decisions, UNVERIFIED) |
| parallel envs | 256 | 8,192 | — | — |
| inference batch per host step | trainee ≤ 256 rows in one bucket + ~20 opponent slots of ~11 rows each (bucket cap 64) | ≤ 8,192 rows (plausibly one network, or a few league nets) | — | — |
| samples per update | 98,304 | 262,144 | — | — |
| epochs per update | **10** | unstated; **≈ 1–2 implied** (§3.1) | — | — |
| minibatch | micro-batch 2,048 × accumulation 32 = **65,536 effective; 2 optimizer steps an epoch, 20 per rollout** | unstated | — | — |
| env steps/s | **1,346** (3080 Ti) | ~100,000 (5090) | — | ~165 games/s (H100) ≈ 2–8k decisions/s |
| env core per CPU core | **6,656 rows/s BUSY** (load 12; quiet records 12–14k): ≈ 75 battles/s busy, ≈ 145 quiet, at a trained game's ~89 rows (both sides) | ~7k "battles"/s | 38k/s per thread | JAX simulator on the accelerator |
| GPU | RTX 3080 Ti | RTX 5090 | RTX 5090 (rented) | H100 |

### 3.1 What their update numbers force

The PPO gradient takes ~70 % of 10 µs = **7 µs per sample**, on a card whose peak dense BF16 tensor rate is at most
209.5 TFLOP/s (NVIDIA's sheet; GeForce parts may run FP32 accumulation at half that, UNVERIFIED for Blackwell). So
their whole learner work per sample is at most:

| their utilisation of peak | ≤ MFLOP per sample, all epochs, forward + backward | ours (2,446 = 10 × 244.6) is |
|---|---|---|
| 100 % | 1,466 | ×1.7 |
| 50 % | 733 | ×3.3 |
| 30 % | 440 | ×5.6 |

A transformer of Wolfe's stated shape needs 100–300 MFLOP forward at 8–24 tokens, i.e. 300–900 per sample per pass
(forward + 2× backward). **That fits the budget only at about 1–2 passes per sample.** We do 10. Our own utilisation is
**5.7 TFLOP/s counted on the 3080 Ti (8 % of its 68.2 dense FP32-accumulate peak) and 12.9 on the 5090 (6.2 % of
209.5)**. Elementwise work is not counted, and it is the bulk of our kernel time: pointwise + reductions are 41 % of
it, GEMM only 36 % (`design_compute_efficiency.md` §1.4).

## 4. Measurements on the CPU (this audit)

### 4.1 The env core's rate per core (`env_core_rate.py` → `result/env_core_rate.json`)

The release cdylib of this checkout (`f7549d9f`), the `ffi_benchmark` shape: the bridge corpus teams, both sides'
rows out, a seeded uniform-random legal policy, N = 48 envs, 6 timed blocks of 300 STEPs. Every row is the full
per-decision work: the Showdown-port step, its protocol emission, both sides' parse chains, and the 2,845-float
observation encode.

| threads | µs per row inside `Core::dispatch` | rows/s | battles/s (random policy) | rows per battle (random policy) |
|---|---|---|---|---|
| 1 | **150.2** | **6,656** | 41.3 | 158.9 |
| 3 (rows/s of the 3 threads together) | 109.3 | 9,149 | 57.6 | 158.9 |

- **BUSY (load average 12.0–13.4 on 16 threads; `utils.contention`: CPU-STARVED, expect ~2× slower).** The quiet-box
  records are 72–82 µs per row at T = 1 (M5 lanes A / B, 2026-09-29), so ×2.0 is the contention, as predicted. It is
  labelled, not corrected.
- T = 3 scales only 1.37× on the agents' 3 physical cores, beside the other agents' jobs.
- The random policy plays long games (159 rows per battle vs a trained game's ~89 rows from both sides, per X26). So
  **a battles/s figure must be converted with the TRAINED length:** ≈ 75 battles/s per core busy, ≈ 145 quiet.
- Python's own share of this loop (actions + staging) is 1.3 µs per row, and the FFI transport ≈ 0.
- **Against theirs:** their "7k battles/s per core" is 48–93× ours if it means whole battles, and about 1–2× if it
  means battle-turns (one turn ≈ 2 of our rows). The survey cannot tell which (F2).

### 4.2 Where the core's time goes (`classify_profile.py` → `result/env_core_profile_classes.md`)

`perf record -F 499` over 3 blocks at T = 1 (6,401 samples), self time by symbol, classified by module:

| class | self time |
|---|---|
| allocation and memory moves (libc `malloc` / `free`, Rust `RawVec` growth, `memmove`) | 36.7 % |
| parse / fold of the emitted protocol into each side's view (`core_events`, `present`, `trackers`, `version`, `dex`) | 24.7 % |
| strings and formatting (`String` builders, `core::fmt`, iterators) | 16.7 % |
| the observation encoder | 9.5 % |
| protocol emission + the bridge (`protocol`, `bridge`, JSON escape) | 4.6 % |
| **the battle simulation itself** (`turn`, `state`, `engine`, `damage`, …) | **2.3 %** |
| other (Rust generic, kernel, Python, the env pool) | 5.8 % |

**Reading.** The env core is a Showdown-FAITHFUL protocol pipeline: simulate, emit text and request JSON, re-parse it
into each side's view, encode. That is by design, because the same chain reads a live websocket game
(`designs/rust_sim/live_reader.md`). Almost all of its cost is the text round trip and the allocator, not the battle
rules.
- **Caveat:** DWARF call-graph unwinding did not resolve callers on the release build (a second profile came back
  self-time only). So allocation and string time are NOT charged to their callers, and the 2.3 % for the simulation is
  a floor, not a share (F3).
- **The implication** is bounded by Amdahl: the core is 7.6 % of X26's cycle.
- **Two cheap descriptors** if it ever matters: a global allocator swap (mimalloc / jemalloc; allocation is 37 % of
  self time), and a sim-only rate with emission off. Neither is measured.

### 4.3 The Python side of collection (X26's own timers, read-only)

Per host step: submit 1.65 + flush 2.37 + opponent draw 0.45 + trainee draw 0.17 + arena write 0.72 + post 0.74 =
**6.09 ms**. That is 25.8 µs per trainee step, 2.53 s per cycle, **3.5 % of the cycle**. `fill` adds 0.27 s per rollout.
- The `flush` call (2.37 ms) only LAUNCHES ~21 slot replays. It is part of the fan-out cost, not Python waste.
- **Not a lever on its own.**

## 5. The GPU measurements for the 2026-10-19 gap (WRITTEN, NOT RUN)

X26 stops at about 380M steps (≈ 10-19 11:00 PDT). The agent granted the GPU lease runs:

```bash
scripts/ops/gpu_lease.sh acquire --owner <agent>           # prints the token
export GEN3AI_GPU_LEASE_TOKEN=<token>
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
designs/research_state/measurements/throughput_audit_2026-10-18/run_gpu_gap.sh /tmp/gpu_gap_2026-10-19
python3 designs/research_state/measurements/throughput_audit_2026-10-18/read_gpu_gap.py /tmp/gpu_gap_2026-10-19
scripts/ops/gpu_lease.sh release --owner <agent>
```

`run_gpu_gap.sh` runs, in order of value per GPU-minute (it is incremental: a cell already `ok` is skipped):

| # | cell | what it answers | ≈ min |
|---|---|---|---|
| 1 | **the frozen suite, every phase** (`python -m agents.training.gpu_bench_suite run --out …/suite_3080ti --tag "RTX 3080 Ti (local, 9800X3D)"`) | fills the PENDING local row of `designs/ops/cloud_gpu_benchmark.md` §0, and gives the E1 / marginal-epoch split on OUR card (today scaled from a 3090) | 40–60 |
| 2 | `u_e10_beff8k`: the bf16 update at micro-batch 2,048 × accumulation 4 (effective 8,192; 12 optimizer steps an epoch) | what the "fewer epochs at the same optimizer-step count" lever pays per extra optimizer step | ~8 |
| 3 | `u_e10_beff2k`: × accumulation 1 (48 steps an epoch) | the same, at the extreme | ~8 |
| 4 | `c_pool1` / `c_pool5`: collection at 256 envs against a 1- / 5-snapshot fixture pool | the ceiling of the fan-out lever (T24: +1.0 ms of GPU wait per host step per snapshot) | 15–20 each |
| 5 | `c_n512` / `c_n1024`: collection at 512 / 1,024 envs (n_steps 192 / 96, the same 98,304 rows), pool 20 | whether a bigger batch amortises the fan-out on the END-STATE network (SIZING measured only the pre-end-state network, 2026-10-01) | 15–20 each |

- **Total:** about 2.5–3 h. Each cell is the suite's own `_phase` worker on `phase_config(...)` with ONE field changed
  (`gpu_gap_cells.py`), so its frozen inputs are the suite's.
- **If the gap is short,** run cells 1–3 and skip 4–5.
- **A sixth, separate item:** T44 (the GPU per-component profile, ≈ 30 min, `designs/ops/TASK_BACKLOG.md`). It sizes the
  edges' share of the update and is its own unit.
- **Caveats each cell records:**
  - The collect cell's pool is untrained fixtures, so compare cells with EACH OTHER, not with X26's absolute.
  - The collect cell SKIPS the update-fit check. A 1,024-env run must still pass the trainer's own fit check on the
    12 GB card before any launch (F5).

## 6. The levers, ranked by expected gain × risk

Gains are on X26's measured 73.0 s cycle (×1.00 = 1,346 steps/s), each ALONE unless stated.

| rank | lever | expected gain | evidence | what it risks | cheapest A/B |
|---|---|---|---|---|---|
| **1** | **Fewer PPO epochs, the optimizer-step count held** (E2 × effective batch 8,192 = 24 steps per rollout, against E10 × 65,536 = 20) | **×1.81** (E2); ×1.64 (E3); ×1.39 (E5) | the suite's per-epoch split (MEASURED on a 3090, scaled); mikumiku's numbers fit ~1–2 passes; PPG (Cobbe 2021) and OpenAI Five reuse ~1; the effective batch is 7–15× the gradient noise scale (SIZING D-10), so smaller batches cost little per step; the toy beat production at EQUAL WALL (+6.5 pp) | **sample efficiency**: E5 at a fresh KL-controlled LR lost −10.56 pp untaught at MATCHED SAMPLES (SIZING, 2026-10-02; dose-confounded, X27). Per WALL it is UNMEASURED. A pre-plateau read is PROVISIONAL | **X44:** fresh `--arch production --n-epochs 2 --grad-accum-steps 4` vs X26's recipe, MATCHED GPU WALL (E10 to 15M ≈ 2.6 h; E2 gets the same wall, ≈ 22–25M steps), 2 seeds per arm as a pilot ≈ **10.4 GPU-h**, then 8 per arm if it is not a clear loss. Two knobs move to hold the step count; if it fails, split it (E2 at 65,536 vs E10 at 8,192) |
| **2** | **Cut the opponent inference fan-out**: (a) cap the snapshots LIVE in one rollout (e.g. 4 of 20, rotated every rollout), or (b) one batched forward over the stacked snapshots' weights | est. **×1.08–1.12** (a), ×1.10–1.20 (b) | T24: GPU wait rises +1.0 ms per host step per snapshot (20.4 ms at 1, 40.2 at 20); inference costs 88 µs/row vs ≈ 14 µs for the same forward inside the learner; SIZING at N = 48: the opponents are 6.69 of 8.63 ms of a flush over 19.5 slots; a grouped forward was capped at 4.7 of 8.4 ms | (a) opponent diversity WITHIN a rollout (the pool's marginal is unchanged across rollouts): low. (b) none to learning (a parity check, at fp32-reassociation tolerance); a T2 build (M) | tomorrow's `c_pool1` / `c_pool5` bound it (no learning run). Then (a) needs a small build plus **X45**: a 2-seed fork pilot at matched wall, ≈ 9 GPU-h; (b) needs only a parity gate |
| **3** | **Overlap collection with the update** (X14) | est. ×1.15–1.3 today (the host CPU's 9.3 s per cycle hides fully; inference partly); **×1.3–1.5 after rank 1**, when collection dominates (E2: update 9.4 s vs collect 26 s) | SIZING's overlapped collector: ×1.24 at 1,024 envs (harness only); the GPU is idle in the host-CPU part of collection | every row one policy version stale (needs IS / V-trace correction); the K9(b) behaviour-consistency machinery; M–L build | X14 as registered; measure its idle-phase budget first (the backlog row's own condition) |
| 4 | **More parallel envs** (512 / 1,024) | est. ×1.03–1.10 at E10; more after ranks 1–2 | SIZING (pre-end-state net): 1,024 envs +9 % on the serial collection rate; 24.4 % predicted staleness at 1,024 | staleness; 12 GB of memory; learning per sample measured only from 48 to 256 envs ("no loss detected", not equivalence) | tomorrow's `c_n512` / `c_n1024`. If either cuts collection ≥ 10 %: **X46**, matched-wall pilot, 2 seeds, ≈ 10 GPU-h |
| 5 | **The network's per-row GPU efficiency** (T44 → T42 edge-bias block assembly; fusing the op / beliefs' elementwise chains) | est. ×1.03–1.10 per fix; unknown in total | our counted utilisation is 6–8 %; pointwise + reductions are 41 % of kernel time; bf16 sped GEMM −28 % but pointwise only −7 % | none to learning if parity holds (not bit-identical: fp32 reassociation) | T44 first (≈ 30 min GPU), then parity-gated builds; no learning A/B |
| 6 | **Restart and eval overheads** | ×1.01–1.05 | T24: a warm 6 h restart costs 1.5 %, a cache-missing one 6.3 % (T47, the cache key); in-loop eval ≈ 3 % | none | T47 (S build); the 24 h interval is the owner's pending decision (T24) |
| 7 | **History seats / model size** | est. ×1.1–1.2 (history ≈ 37 % of FLOPs, ≈ 20 % of the GPU update, ESTIMATED; also inference) | `design_compute_efficiency.md` §1.3–1.5 | **strength: UNKNOWN** (never isolated). The owner's 2026-10-16 framing rejects slimming production | none now. X36 (deferred) is where it would be measured |
| 8 | **More mixed precision** (bf16 rollout inference X19; bf16 beyond the trunk) | ×1.02–1.05 | inference is latency-bound, not arithmetic-bound; pointwise gains −7 % under bf16 | opponent / behaviour fidelity (X19's ±2 pp bar); decisions stay fp32 by policy | X19 as registered |
| 9 | **Env-core speed** (allocator, a native-state encoder) | ≤ ×1.08 even at infinite speed; ~×1.02 for an allocator swap | §4.1–4.2: 7.6 % of the cycle; 37 % of its self time is allocation | a native encoder would fork the live-reader chain (the parity it buys) | none needed now |
| — | **Larger micro-batches** | none on our card | the suite: ≤ 7 % on a 3090, SLOWER on a 600 W 5090; 4,096 does not fit 12 GB | — | — |

**Why rank 1 first, despite its risk.** It is the only lever above ×1.2 that needs no build, and its A/B is one typed
flag pair. Its result also changes the order of the rest: after it, collection is the larger half and ranks 2–4 grow.

**Interactions (ESTIMATED, sequential on the 3080 Ti):**

| step | cycle |
|---|---|
| E10 (today) | 73.0 s |
| E2 | 40.4 s |
| + the fan-out cut | ~33–35 s |
| + overlap | ~24–30 s |

That is **×2.4–3.0, ≈ 3,300–4,100 steps/s**, and ≈ 5,500–7,000 with the 5090 on top. The ~15× that remains against
mikumiku's face value is the network's per-row GPU cost plus the units question (F1).

## 7. FINDINGS (standing rule 7)

- **F1 (their units).** mikumiku37's stated figures disagree with each other: 330M games in 48.3 h with 100k steps/s
  implies 52.7 steps per game, where a Bo1 VGC game lasts ~6–12 turns. Either "steps" counts both sides × both slots,
  or one figure is a peak. So the per-decision gap is ~15–75×, not a firm 75×. UNVERIFIED either way (a forum
  self-report read through a summariser).
- **F2 ("battles/s").** Neither "7k battles/s per core" (mikumiku37) nor "38k battles/s per thread" (Wolfe_Alt6767)
  says whether a battle is a whole game or one turn, so the env-core comparison spans ~1–500× (Wolfe's per-turn reading
  ≈ 6–12×, per-game ≈ 260–500×).
- **F3 (the profile is self time only).** DWARF unwinding on the release cdylib did not resolve callers (two profiles;
  the call-graph one returned leaf frames only). Allocation and string time are therefore not charged to their callers,
  and the simulation's 2.3 % is a floor. A frame-pointer or debuginfo build would close it.
- **F4 (busy box).** Every CPU number here was read at load average 11–13 on 16 threads, on the agents' 3 physical
  cores, beside the maturity battery and the team-strength read: 150 µs per row against the 72–82 quiet records.
  WARN, never stretched.
- **F5 (1,024 envs: memory UNVERIFIED).** The suite's collect cell skips the trainer's update-fit check. Tomorrow's
  `c_n1024` reads collection only. A launch at 1,024 must still pass the fit check on the 12 GB card (SIZING: 256 fits
  with 2.2 GiB of headroom).
- **F6 (the epoch evidence is dose-confounded and per-sample).** The one fresh-run epoch read (E5 −10.56 pp) matched
  SAMPLES, not wall, and not dose (X27). X28's epoch controller is the owner's named epoch lever after X26. X44 is the
  cheaper wall-matched read and does not replace X28.
- **F7 (the 5090 factor).** ×1.75 is the cloud bench's Amdahl rule re-applied to X26's current cycle (the bench quotes
  ×1.70 at 258.6M). It assumes our 3080 Ti equals the reference 3090, which stays UNVERIFIED until §5's cell 1 lands.
- **F8 (their peak).** The 5090's 209.5 TFLOP/s dense BF16 is NVIDIA's sheet figure. If GeForce Blackwell runs FP32
  accumulation at half rate, their budget in §3.1 halves and the "≈ 1–2 passes" inference only tightens. UNVERIFIED.
- **F9 (a backlog build not filed).** Lever 2(a), the live-snapshot cap, needs a small build (opponent-route sampling
  per rollout). It is not yet a `TASK_BACKLOG.md` row. This unit added the experiment row (X45) only, per its brief.

## Files

| file | what |
|---|---|
| `gap_decomposition.py` → `result/gap_decomposition.txt` | every input with provenance; the per-step budget, the card, the update ceiling, the epoch lever |
| `param_count.py` | our learner's parameters at HEAD (CPU) |
| `env_core_rate.py` → `result/env_core_rate.json` | the env core per core: rows/s, battles/s, rows per battle |
| `classify_profile.py` → `result/env_core_profile_classes.md`, `result/env_core_profile_top60.txt` | the perf self-time classification |
| `gpu_gap_cells.py`, `run_gpu_gap.sh`, `read_gpu_gap.py` | tomorrow's GPU cells, their runner (needs a lease) and their reader |
