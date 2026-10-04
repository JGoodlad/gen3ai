# X5 cost against the budget, and which pieces carry it (2026-10-04)

The cost pass after X5's core build unit U3 (`designs/endstate/design_x5_belief_tokens.md` §3.6, F-X5-30), run
under the GPU lease (owner `x5-cost`). It measures `--belief-tokens fixed_mass` against the production `blob`, both
**with the X26 ride-along heads ON** (the real A/B configuration), then ablates the X5 pieces on the extractor's
compiled forward + backward. Nothing in the shipped code was changed. Tags: **MEASURED** unless marked.

- **Commit measured: `889add9d`** (U3 part 3b; `origin/main` when the agent started; the worktree stayed on it).
  **Landed after it and NOT measured:** U4 (`350fb83b`: the flat pointer, OTHER_move priced on the full move axis —
  more X5 work, F-X5-37) and U6's F-X5-44 fix (`61be9ec9`: "a DEAD OTHER's uniform move order counted as rule-8 ties:
  47 % excluded" — the K9(b) tie share of §1 item 4).
- **Box.** RTX 3080 Ti (12 GiB), torch 2.8.0+cu126, fp32, the production recipe at N = 256 (98,304 rows per
  rollout, batch 2,048 × accumulation 32, 10 epochs = 480 micro-batches per update), Rust env core, T2 graph
  backend, buckets (8, 64, 256), 8 lanes.
- **Every launch:** `--arch production --device cuda --steps 6000000 --eval-freq 50000000` plus
  `--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all` (the X26
  argv, `ridealong_baseline/PREREGISTRATION.md`); the fixed_mass arm adds `--belief-tokens fixed_mass
  --allow-nonproduction-arch`. `checkargs`: the only ARCH-surface drift is `belief_tokens`, every RECIPE knob matches.
  Eval off, so no eval cycle and no ladder child runs inside a measured window.
- **Two regimes.** **B** = an empty pool: the trainee plays the in-core bots, T2 serves the trainee only. **A** = the
  production steady state: a seeded 20-snapshot pool (`scripts/build_pool.py`: the arm's own 1.2M-step checkpoint
  from its regime-B launch, perturbed with 20 seeds; `summary.json` win rate 0.94), self-play 90 % from the first
  rollout (`rust_env/p2_policy_decisions` ≈ 93k of 98k opponent decisions per rollout in all four regime-A launches),
  T2 serving the trainee + 20 pool slots.
- **Launches**, in order: blobB, fmB (refused, then re-run), blobA1, blobA2, fmA3, fmA4 (fmA1, fmA1b and fmA2 were
  refused at startup or update 2 — §1). Launched through the bottleneck profile's phase-probe wrapper
  (`scripts/phase_probe_run.py`, the trainer's own `main()` unchanged, its script name kept out of the argv) by
  `scripts/drive.sh` (CPU sampler, `nvidia-smi` at 2 Hz, SIGTERM to the trainer's PID after 13 update rows).
- **Quiet updates only** (standing rule 8): an update whose window holds any CPU-sampler row with a windowed
  contention factor ≥ 1.05 is excluded, as are the dry update, the first real update and the compile-canary
  update; every excluded update is listed in `results/<launch>.json`. Other agents' gates ran during most launches,
  so the kept counts are uneven (blobB 8, fmB 7, blobA1 5, blobA2 3, fmA3 2, fmA4 9).

## Verdict against the pre-registered budget (§3.6)

| budget line | blob + X26 | fixed_mass + X26 | Δ | budget | verdict |
|---|---|---|---|---|---|
| **`train_ms`** (update wall, quiet median; regime B) | 40.96 s [40.94, 40.99] | 47.80 s [47.54, 47.96] | **+6.84 s = +16.7 %** (CI of the difference +16.0 to +17.1 %) | ≤ +5 % | **OVER (3.3×)** |
| **T2 service per host step**, regime A (flush + GPU wait, every slot) | 21.76 ms (21.79, 21.73) | 29.60 ms (29.78, 29.41) | **+7.83 ms = +36.0 %** | ≤ +3 % | **OVER (12×)** |
| T2 service per host step, regime B (trainee only) | 4.90 ms | 6.59 ms | +1.69 ms = +34.4 % | — | (same direction) |
| **D-6 headroom** at N = 256 with the X26 heads (`UpdateFit`, regime A) | 2,074 MiB | **432 MiB** | **−1,642 MiB** | ≥ 1,024 | **FAIL — the launch is REFUSED** (`UpdateWontFit`) |
| D-6 headroom, regime B | 2,046 MiB | 556 MiB | −1,490 MiB | ≥ 1,024 | FAIL (refused) |
| startup, start → first rollout (regime A, two launches each) | 472, 418 s | 689, 846 s | +323 s (+72 %) | — | T2 build +237 s, R1 prewarm +66 s |
| rollout (play) wall per cycle, quiet | B 5.64 s · A 13.70 s | B 7.96 s · A 17.12 s | B +41 % · A +25 % | — | not a budget line; enters §7's wall-clock rule |

**All three budget lines fail; the U8 rule applies: STOP before any A/B GPU.** And four separate startup / early
refusals (§1) mean a fixed_mass production launch does not currently run at all without overrides.

- **`train_ms` is read in regime B.** The update is the same program in both regimes (blob: 40.96 s in B, 40.87 s in
  A, Δ −0.2 %). fixed_mass's regime-A update reads 54.3 s [53.95, 55.54] because both regime-A launches ran with
  `--behaviour-check warn` (§1, refusal 4), and the K9(b) violation path then runs a full-buffer scan inside every
  update (13 of 13 updates of fmA3 log it; fmB logs it once, at the dry update). That +6.5 s is the warn path, not X5.
- **"T2 flush" defined (F-G-3 resolved).** The budget's base, "13.12 ms (trainee 4.48 ms; opponents
  dispatch-bound)", names no TB tag. The collector pays T2 per host step as `rust_env/flush_ms_per_host_step` (the
  host's graph launches for every slot) + `rust_env/gpu_wait_ms_per_host_step` (the wait for every slot's results);
  the sum is the whole T2 cost of a host step, and in regime B (trainee only) it is the trainee's own share (4.90 ms
  here, matching the note's 4.48 ms trainee figure within the box's drift). This README reads the budget line as
  that SUM in regime A, the steady state X26 and the A/B live in. Its components: host flush +0.26 ms (+13.8 %), GPU
  wait +7.57 ms (+38.1 %). The production regime-A total here (21.8 ms) matches the bottleneck profile's 21.5 ms, not
  the note's 13.12 ms, whose provenance is not recorded (sizing B, before the deletion pass, is the likely source —
  UNVERIFIED).
- **Memory, where the 1.5–1.6 GiB goes** (the startup ledger, regime A means; reserved in brackets):

  | stage | blob | fixed_mass | Δ |
  |---|---|---|---|
  | rust env core (T2 slots, staging, arena) | 920 [2,250] MiB | 1,261 [2,956] | **+341 [+706]** |
  | compiled regions (gate + prewarm + lock) | +201 [+4,660] | +479 [+4,942] | **+278 [+282]** |
  | the first update's peak (`UpdateFit`): allocated / reserved | 6,430 / 8,126 | 7,162 / 9,804 | **+732 / +1,678** |
  | steady update peak allocated (TB) | 4,843 | 5,264 | +422 |

  The trainer's floor (weights + optimizer + lifecycle) is 1,144 → 1,765 MiB. U3 part 1 alone cost −180 MiB of headroom
  without the X26 heads (`x5_u2_gpu_checks_2026-10-04`); the X26 heads cost ≈ −100 (blob 2,143 → 2,046). So U3 parts 2–3
  cost ≈ −1.3 GiB, in T2 (+706 reserved) and the update's activations (+690 reserved above the startup reserve).

## 1. Four refusals a fixed_mass production launch hits today (each a FINDING)

`results/refused_launches_excerpts.txt` holds each one's log lines.

1. **`UpdateWontFit` at the dry first update** (fmB, first try; every fixed_mass launch would): headroom 556 MiB
   (regime B) / 432 MiB (regime A) vs the declared 1,024. Every later fixed_mass launch here ran with
   `PROF_FIT_HEADROOM_MIB=256` — a measurement-only lowering of `update_fit.FIT_HEADROOM_MIB` in the wrapper's process;
   no OOM, no allocator retry (`lifecycle/cuda_alloc_retries` 0) in any measured update.
2. **`LazyAcquisitionError` at the first pool load** (fmA1, regime A): "loading pool:20000 into T2 slot 1 left 1.1 MiB
   NEWLY allocated on cuda" against the 1 MiB tolerance (`rust_rollout/build.py` `SLOT_LOAD_ALLOC_TOLERANCE`). The blob
   pool of the same construction loads cleanly. So **a fixed_mass run cannot serve its own self-play pool**: a real run
   would die at its first pool refresh. Cause UNVERIFIED (each `svc.load` runs the eager parity gate at every bucket;
   something on the fixed_mass eager path allocates ≈ 1.1 MiB per load). The regime-A fixed_mass launches ran with
   `PROF_SLOT_LOAD_TOL_MIB=8` (measurement only).
3. **`[CompileSentinel] FATAL`: compiled TRAIN gradient vs eager, cosine 0.000000** (fmA1b, regime A; same argv and
   pool as fmA3 / fmA4, which passed). One of four fixed_mass launches that reached the gate. Related: in EVERY
   fixed_mass gate and canary here the compiled-vs-eager difference reads **exactly 0.00e+00** on the loss and the
   train features (blob reads 2.34e-07 / 2.38e-07 at the same gate), which a real Inductor graph does not produce —
   the gate's two sides may not be independent under fixed_mass. UNVERIFIED; GIGO risk on the gate itself.
4. **K9(b) `BehaviourMismatch` (FATAL) at the first real update** (fmA2, regime A): `excluded_frac` 0.573 ≥ the 0.15
   ceiling — 1,174 of 2,048 rows sit within the relative margin 0.0002 of a selection cutoff, the cutoff named as
   `hypothesis_set.py:193 argsort` (the stable order); the largest |Δ log π| is 0 (no real mismatch). Regime A reads
   0.54–0.58 every update; regime B 0.09–0.14 (passes, close to the line; blob 0.02–0.07). The dry update's fixture
   reads 0.45–0.47 in every fixed_mass launch. **A fresh fixed_mass self-play run dies here by design** until the tie
   margin or the ceiling is re-measured for X5 (`k9_behaviour_exclusion/result.json` is the ceiling's evidence).
   fmA3 / fmA4 ran with `--behaviour-check warn` (a real flag). **Since fixed on main, after the measured commit:**
   `61be9ec9` (F-X5-44) names exactly this — a dead OTHER's uniform move order counted as ties, 47 % excluded — which
   matches the 0.45–0.47 dry-update reads here; whether it also clears regime A's 0.54–0.58 is UNVERIFIED.

## 2. The launches

| | blobB | fmB | blobA1 | blobA2 | fmA3 | fmA4 |
|---|---|---|---|---|---|---|
| quiet updates (median s) | 8 (40.96) | 7 (47.80) | 5 (40.85) | 3 (40.98) | 2 (54.87)* | 9 (54.30)* |
| T2 host flush / GPU wait (ms per host step) | 0.56 / 4.34 | 0.71 / 5.88 | 1.87 / 19.92 | 1.88 / 19.85 | 2.31 / 27.47 | 1.96 / 27.45 |
| Rust core step (ms per host step) | 6.56 | 10.00 | 7.75 | 7.94 | 11.68 | 8.42 |
| T2 build / R1 reset + prewarm (s) | 252 / 125 | 428 / 175 | 280 / 135 | 245 / 118 | 444 / 177 | 555 / 209 |
| startup windows contended | 15 / 198 | 37 / 314 | 58 / 234 | 1 / 207 | 19 / 342 | 208 / 420 |
| `UpdateFit` headroom (MiB) | 2,046 | 556 | 2,074 | 2,074 | 432 | 432 |

\* inflated by the K9(b) warn path (refusal 4); not an X5 cost. Graphs: R1 4 in process, `recompiles_after_lock` 0 in
every launch; the compile canary at update 10 passed in every launch that reached it.

- **The startup** grows by about 4–6 minutes: T2's build +176 s (B) / +237 s (A), R1's reset + prewarm +51 / +66 s.
  The T2 build moves ±55 s between same-arm launches under contention (F-G-7), so read these as ±1 min.
- **The Rust core step** rises 6.6 → 10.0 ms (B) and 7.8 → 10.0 ms (A) per host step on quiet rollouts. The env core
  does not run X5; the cause is UNVERIFIED (the collector overlaps the core step with T2, so a longer GPU forward may
  surface there). With T2, it makes the rollout +41 % (B) / +25 % (A).

## 3. Ablation: which X5 pieces carry the cost

**Method** (`scripts/ablate.py`). The production-surface extractor (`production_args()`, fixed_mass for every arm
but `blob`), the gates' seeded perturbation, `torch.compile(fullgraph=True, dynamic=False)`, forward + backward of
`pi.sum() + vf.sum()` on 2,048 real rows (`gen_real_obs.py`: the Rust collector, complete games) — the U2 checks'
`compile_ab.py --bwd`, the learner's micro-batch shape. One fresh process per arm and replicate; median of 5 blocks × 20
steps; replicate 1 in arm order, replicate 2 reversed. Each arm removes ONE piece by making its result dead or
constant (a monkeypatch in that process only — never a shipped change, and what it computes is NOT the design), so
Inductor drops it and its backward. **T2 columns**: the same forward under `no_grad`, compiled per batch, CAPTURED in
a CUDA graph and timed by replay (5 × 200), at T2's buckets — replicate 1 only. This is subtraction, not hardware
counters (admin-blocked on this boot; F-BP-5).

**The X5 increment on this harness: 72.73 → 89.23 ms per micro-batch (+16.50 ms, +22.7 %; ×480 = 7.9 s per update,
against the 6.84 s measured in the launches — the harness is the extractor alone, in eval mode, without the aux
losses and the optimizer, so its SHARES are the finding and its absolute per-update figure is an upper estimate).**
T2 forward: +0.62 ms at B = 8, +0.72 at 64, +1.47 at 256 (+51 %, +38 %, +36 %; the B = 256 figure matches the
launches' trainee wait, +35 %).

| piece removed (arm) | train: ms saved per micro-batch (reps) | share of X5 | ≈ s per update | T2 saved, ms per forward at B = 8 / 64 / 256 |
|---|---|---|---|---|
| **the hypothesis `PokemonEncoder` pass** (`no_hypenc`) | **9.67** (9.39, 9.94) | **58.6 %** | 4.64 | −0.04 / 0.07 / 0.29 |
| **everything else (the residual: `all_off` vs `blob`)** | **3.74** | **22.7 %** | 1.80 | **0.27 / 0.29 / 0.55** |
| every fixed-size construction's 64 bisection steps (`no_bisect`) | 0.84 (0.80, 0.88) | 5.1 % | 0.40 | 0.09 / 0.10 / 0.16 |
| every stable argsort (`no_argsort`) | 0.80 (0.71, 0.89) | 4.8 % | 0.38 | 0.05 / 0.06 / 0.08 |
| **OTHER's second op pass** (`no_other`) | 0.68 (0.77, 0.58) | 4.1 % | 0.33 | **0.21 / 0.21 / 0.22** |
| the K+16 seat axis (`no_ext`) | 0.58 (0.68, 0.48) | 3.5 % | 0.28 | 0.01 / 0.00 / 0.05 |
| the per-mon move construction + its order (`no_permon`) | 0.02 | 0.1 % | 0.01 | 0.00 / 0.00 / 0.00 |
| the kept hypothesis-attacker work (`no_hypatk`) | 0.03 | 0.2 % | 0.01 | 0.00 / 0.00 / 0.00 |
| the trunk's log-π key bias (`no_keybias`) | −0.31 (−0.44, −0.18) | ≈ 0 | — | 0.00 / 0.00 / 0.02 |
| all six above at once (`all_off`) | 12.76 (12.97, 12.54) | 77.3 % | 6.12 | 0.35 / 0.43 / 0.92 |

- **Noise.** fm's two replicates agree to 0.04 ms; blob's differ by 1.7 ms (71.88, 73.58), so the X5 total carries
  about ±0.85 ms and every row under 1 ms is a ranking, not a precise value. The T2 replay figures are stable to
  ~0.01 ms but are one replicate.
- **What the residual is** (torch.profiler, 3 steps, `results/ablate_{blob,fm,all_off,no_hypenc,no_other}_r2.json`):
  per step blob 71.2 ms / 1,491 kernel launches, fm 88.9 / 1,854, all_off 75.6 / 1,644. The hypothesis encoder pass is
  GEMM (−7.35 ms) + pointwise (−2.58), 127 launches. The residual over blob is POINTWISE (+5.16 ms) and launches (+153),
  GEMM only +1.08: elementwise work — δ_θ, the hypothesis set's gathers, the T0 belief heads on the hypothesis context,
  the op's roster gates, OTHER's trunk seat, the pools' float masks — not attributed further here (TODO, a finer
  ablation or the GPU counters after the reboot). At T2's small batches the residual and OTHER's pass are most of X5's
  cost (B = 8: 0.27 + 0.21 of 0.62 ms), because there the cost is kernel count, not FLOPs.
- **The per-mon construction, the attacker gating and the key bias are free** (dense kernels; gating multiplies). The
  64 unrolled steps stay cheap (U2's F-X5-22 verdict holds with five constructions in the graph).
- **Memory per piece is not readable here**: the harness peak (eval mode, extractor only) moves +52 MiB blob → fm
  against +422 MiB steady / +732 MiB dry-update allocated in the launches; the launches' ledger (above) is the memory read.

## 4. Optimization proposals, ranked by expected saving

Not implemented (the orchestrator dispatches after the owner sees these). "Exact" = computes the same values, up to
fp32 reassociation where marked. Savings are this harness's (training: per micro-batch × 480; T2: per forward).

| # | change | semantics | expected saving | risk |
|---|---|---|---|---|
| **P1** | **Run the hypothesis `PokemonEncoder` pass over the 6 OPPONENT slots only**, not all 12. The encoder is per-mon (its only cross-mon input is the active-context scatter, and a hypothesis is never the active; `hypothesis_tokens` docstring), and the pass's 6 own-team rows are discarded today | exact | ≈ half of 9.67 → **≈ 4.8 ms ≈ 2.3 s per update (≈ −5.6 pp of the +16.7 %)**; ≈ 0.15 ms per T2 forward at B = 256; halves that pass's saved activations (D-6) | low–medium (the encoder's [B,12] layout, the `last_move_tokens` side effect, the active-context scatter) |
| **P2** | **Attribute the residual before optimising it** (3.74 ms, 22.7 %; the largest T2 piece at every bucket): a finer ablation of δ_θ, OTHER's trunk seat, the T0 heads on the hypothesis context, the hypothesis set's gathers, the pools' float masks — or the GPU counters after the reboot | measurement | unknown, up to 3.7 ms train and 0.27–0.55 ms per T2 forward | none |
| **P3** | **OTHER as a 7th opponent column of the MAIN op pass** (one kernel run over [·, 7] instead of a second six-column run read at one column; F-X5-30's "one-column kernel" is the same idea), and inside today's OTHER pass pass `base=` its own D1 cells to `pairwise_boost` (the main pass already does; the OTHER pass recomputes D1's world — one of its six `_outgoing_matrix` runs) | exact if every opponent column is independent in those kernels (the kernel contract says per-(seat, mon); VERIFY per kernel) | ≈ 0.6 ms train (0.3 s per update); **≈ 0.2 ms per T2 forward at EVERY bucket — 34 % of X5's B = 8 cost — paid by each of the 21 regime-A slots** | medium (six kernels, the roster swap, the D4 / V gates) |
| **P4** | **Fuse each construction's 64-step bisection into ONE GPU kernel** (a Triton kernel looping in registers, registered as a custom op with a fake impl so `fullgraph` holds). Not U2's eager "τ outside the graph", which was slower | exact in exact arithmetic; the per-step sum order may differ (τ at ~1e-7) — a NUMERICS change: re-run U2's 27-comparison parity suite; selection near-ties are already rule-8 excluded | ≈ 0.8 ms train (0.4 s per update); 0.09–0.16 ms per T2 forward | medium |
| **P5** | **Replace the full stable argsort by `topk` on a packed key** (fp32 π bits made monotone, then the num as the tiebreak): consumers read only the first K seats and "rank ≥ K" membership | exact if the packing is exact (ties to the lower num preserved); the stable-tie tests pin it | ≈ 0.8 ms train (0.4 s per update); 0.05–0.08 ms per T2 forward | medium (`move_rank` feeds the bench E5 tail; the species order) |
| **P6** | **The K+16 axis only where it is read**: the 16 typed-HP columns price only the revealed-HP seat, and `mix` is the identity elsewhere — price the K seats plus the 16 typed columns once, without the [K, K+16] contraction over every seat | exact | ≈ 0.6 ms train (0.3 s per update); ≈ 0.05 ms per T2 forward | medium |
| P7 | The per-mon construction, the attacker gating, the key bias | — | ≈ 0 measured | no action |

- **Even all of P1 and P3–P6 do not reach the budget**: ≈ 7.6 of 16.5 ms → X5 ≈ +8.9 ms per micro-batch ≈ +12 % of the
  extractor, ≈ +9 % `train_ms` against +5 %. Reaching +5 % needs the residual (P2) or a DESIGN change for the owner,
  e.g. a cheaper hypothesis token than the full `PokemonEncoder` pass (species embedding + a dex-row MLP without the
  move sub-encoder): up to the whole 9.67 ms, but it changes what is computed.
- **D-6 (memory) is a separate problem** (−1.6 GiB; the floor needs +592 MiB back in regime A): P1 halves the encoder
  pass's saved activations; activation checkpointing of that pass is exact (deterministic recompute) at about its
  forward cost again; T2's graph pools (+706 MiB reserved) answer to the trainer's own numerically-identical fit levers
  (`--t2-lanes`, `--t2-buckets`) at a T2-throughput cost. How much each frees is UNVERIFIED (not measured here).
- **T2 is kernel-count-bound**, so at small batch the order is P3 > P2 > P4 > P5, and P1 matters only at B = 256.

## Findings (standing rule 7)

- **F-XC-1 (budget; STOP).** All three §3.6 lines fail at `889add9d` with the X26 heads: `train_ms` +16.7 % (≤ +5 %),
  T2 service +36.0 % in regime A (≤ +3 %), D-6 headroom 432 MiB (≥ 1,024). The U8 rule: no A/B GPU.
- **F-XC-2 (D-6, a launch blocker).** A fixed_mass `--arch production` launch with the X26 heads is REFUSED at its dry
  update (`UpdateWontFit`, 556 / 432 MiB). U3 parts 2–3 cost ≈ 1.3 GiB of headroom (part 1: 180 MiB).
- **F-XC-3 (a self-play blocker).** Loading a fixed_mass pool snapshot into a T2 slot leaves 1.1 MiB newly allocated
  (`LazyAcquisitionError`, tolerance 1 MiB): a fixed_mass run dies at its first pool refresh. Cause UNVERIFIED.
- **F-XC-4 (GIGO risk on the compile gate).** One fixed_mass launch of four FATALed at the compile sentinel with grad
  cosine 0.000000 (same argv and pool as two that passed), and EVERY fixed_mass gate / canary reads compiled − eager
  exactly 0.00e+00 on the loss and the features (blob 2.3e-07): the gate's two sides may not be independent under
  fixed_mass. Needs a root cause before any fixed_mass run trusts the gate.
- **F-XC-5 (a K9(b) blocker).** fixed_mass rows sit at the stable-order selection cutoff far more often than the K9(b)
  ceiling allows: `excluded_frac` 0.54–0.58 per update in regime A (FATAL at the first real update), 0.09–0.14 in
  regime B (under 0.15, close), 0.45–0.47 on the dry update's fixture; blob 0.02–0.07. The tie margin (relative 2e-4)
  or the ceiling needed an X5-specific look — `61be9ec9` (F-X5-44, after the measured commit) fixes the dead-OTHER tie
  count that matches the 0.47; regime A after that fix is UNVERIFIED. Under `--behaviour-check warn` the violation path's full-buffer scan
  adds ≈ 6.5 s to EVERY update.
- **F-XC-6 (rollout).** The Rust core step per host step rises 6.6 → 10.0 ms (B) and 7.8 → 10.0 ms (A), and the
  rollout wall +41 % (B) / +25 % (A), though the env core runs no X5 code; cause UNVERIFIED (overlap with T2 is the
  likely one). It enters §7's wall-clock rule.
- **F-XC-7 (the harness).** The ablation's absolute per-update figure (7.9 s) exceeds the launches' 6.84 s: the harness is
  the extractor alone in eval mode. Its shares are the finding. Blob's replicates differ by 1.7 ms, so rows under 1 ms
  are rankings. The residual (22.7 %) is not attributed (TODO).
- **F-XC-8 (T2 definition, F-G-3 closed).** The budget's "T2 flush" is read as flush + GPU wait per host step over every
  slot, in regime A. The note's 13.12 ms base has no recorded provenance; this box's production regime A reads 21.8 ms
  (the bottleneck profile: 21.5).
- **F-XC-9 (X26 heads).** On blob the X26 heads cost ≈ +1 s `train_ms` (40.96 here vs the U2 checks' 39.97 without them,
  one commit apart — UNVERIFIED as a clean pair) and ≈ −100 MiB headroom.
- **F-XC-10 (coverage).** Contention from other agents' gates excluded most updates in fmA3 (2 kept) and blobA2 (3);
  the regime-A fixed_mass `train_ms` is the K9(b)-warn-inflated read and is not used for the budget. Startup figures
  carry ±1 min (F-G-7). GPU counters were not available (admin-blocked; the owner's reboot enables them).
- **F-XC-11 (incidents, no damage).** The first driver matched the `gpu_lock.sh` wrapper's PID instead of the trainer's
  (fixed before any stop: the trainer was stopped by its explicit PID); the first orphan check matched this agent's own
  tool shell and SIGTERMed it twice (fixed: the trainer's descendants are recorded before the stop). No `snapshot_ladder`
  child was spawned (eval off); no orphan outlived a launch.

## Reproduce

From a worktree at `889add9d` with `export PYTHONPATH=$PWD/src` and the `gen3ai_torch28` interpreter; every GPU
command under the lease (`GEN3AI_GPU_LEASE_TOKEN_FILE=…`) through `scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh <GB>
timeout <s> …`. Scratch (not committed) sat under the job directory: `S=<scratch>`, `GEN3AI_MODELS_DIR=$S/models`.

```bash
# real rows for the ablation harness (CPU, ~6 s)
python scripts/gen_real_obs.py --out $S/real_obs.npz --envs 64 --steps 64
# one launch (drive.sh: samplers + the phase-probe trainer + SIGTERM by PID at 13 update rows)
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
S=$S W=$PWD PY=… scripts/drive.sh blobB 13 --arch production --device cuda --steps 6000000 --eval-freq 50000000 $X26
PROF_FIT_HEADROOM_MIB=256 … scripts/drive.sh fmB 13 … $X26 --belief-tokens fixed_mass --allow-nonproduction-arch
# regime A: a 20-snapshot pool from the arm's own regime-B checkpoint, then the launch into that run dir
python scripts/build_pool.py --template $S/models/fmB/final_model_interrupted.zip --out $S/models/fmA3/snapshots
PROF_FIT_HEADROOM_MIB=256 PROF_SLOT_LOAD_TOL_MIB=8 … scripts/drive.sh fmA3 13 … --behaviour-check warn
python scripts/read_run.py --tag fmA3 --dir $S --out results/fmA3.json        # per launch
# the ablation arms, one process each
GEN3AI_TEST_ALLOW_GPU=1 python scripts/ablate.py --arm <arm> --obs $S/real_obs.npz --batch 2048 \
    [--t2-batches 8,64,256] [--profile] --out results/ablate_<arm>_r<k>.json
python scripts/summarize.py            # -> results/numbers.json
```

## Files

- `scripts/`: `drive.sh`, `phase_probe_run.py` (the bottleneck profile's wrapper + two measurement-only env overrides),
  `build_pool.py`, `read_run.py` (the U2 checks' reader, its TB tag list extended), `summarize.py` (registered in
  `src/measurements_readout_gate_test.py`), `ablate.py`, `gen_real_obs.py`.
- `results/`: `numbers.json`, `<launch>.json` + `<launch>.drive.log` for the six measured launches, `raw/<launch>.{phases,
  cpu}.jsonl`, `ablate_<arm>_r{1,2}.json`, `refused_launches_excerpts.txt`.
