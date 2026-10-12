# The bf16 STRENGTH A/B: `--trunk-precision bf16` against fp32, on the end state (REGISTERED 2026-10-12, before any seed)

**Status: REGISTERED 2026-10-12, before any seed** (lifted from
[`perf_phase_2026-10-10/PLAN.md`](../research_state/measurements/perf_phase_2026-10-10/PLAN.md) §2d; orchestrator's
call 2026-10-12, option (i)). Run by the Training session. Nothing here has read a cell.

## 1. The question

Is the end state trained with a **bf16 trunk** not detectably worse than the same end state trained **fp32**, at the
closing test's own meter and rule? The bf16 trunk is a numerics change of the TRAINING signal (window D: the
micro-batch gradient 1–18 % rel-L2 from fp32, F-PD-2), so speed never adopts it (PLAN §2d: MANDATORY before adoption).
Speed (the pooled quiet `train_ms` ratio) is REPORTED, never decides.

## 2. The pin, the environment, the arms

- **P_bf16 = `f45d0c2797b892292355dc8b784bb985099f10c1`** (origin/main HEAD at registration). Every seed of BOTH arms
  trains at P_bf16 (`--pin-commit`), and every cross is played there. `--arch endstate` is typed explicitly, so a later
  change of what `--arch production` means does not move either arm. At P_bf16, `checkargs` resolves the arm to
  `endstate@production_config@49cd523f38ee+overlay@2b359668`, the closing test's end-state tag.
- **Environment:** `gen3ai_torch28` (torch 2.8.0+cu126) for the whole experiment, both arms on one build. A CUDA 12.8
  environment being prepared elsewhere is NOT used by any seed or cross here. `GEN3AI_SHARED_COMPILE_CACHE=0` for every
  launch (the closing test's setup).
- **CPU split** (orchestrator, 2026-10-12, arm-neutral): the TRAINING tree (launcher, trainer child, the Rust env core
  and its workers, the eval core) is pinned to physical cores 0–4 (`taskset -c 0-4,8-12`; CPU n and n+8 are SMT
  siblings), the agents' CPU work to 5–7 (`5-7,13-15`), so neither wanders onto the other's cores. In force from
  `rb_bf_eb_s2001`'s first minutes (10:53, applied to its live tree, all threads) and the launch default for every
  later seed and resume; `rb_bf_e_s2001` ran unpinned (affinity 0–15; contention factor p90 1.82). Affinity changes
  only speed, never the training signal, so it is reported, never a validity condition.
- **Arms** (the closing test's recipe, ride-along heads and launcher; only `--trunk-precision` differs):

| arm | argv (each seed adds `--seed 200N --run-name <run>`) | runs |
|---|---|---|
| **E** (fp32) | `--arch endstate --steps <S> --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda` | `rb_bf_e_s2001`–`s2008` |
| **EB** (bf16) | the same `+ --trunk-precision bf16` | `rb_bf_eb_s2001`–`s2008` |

  Launcher: `python -m main.launcher --restart-interval-hours 6 --pin-commit <P_bf16> <argv>`. Validated at P_bf16 on
  2026-10-12 for seed 2001 of each arm: `checkargs` → 0 unrecognized, ✓ ARCH surface = the mirror + arm `endstate`,
  ✓ RECIPE, ✓ still launches; `--dry-run` → ✓ DRY RUN, FRESH, pin `f45d0c27`.

- **Why E is RE-RUN rather than reused** (the identity precondition of PLAN §2d failed, 2026-10-12): `--trunk-precision`
  does not exist at the closing test's P_end `62085750`, a cherry-pick of `50a807b5` onto it conflicts (it depends on the
  earlier perf commits), and the earliest main commit with bf16 + the recorded-version compat fix (`c1fae513`) changes
  the end state's fp32 forward (`5eef9934`, `--ko-ramp exact` as the closed form). Re-running E at P_bf16 also yields
  **eight fresh reference seeds of the end state at HEAD**, the architecture being adopted as production, which the
  next experiments need anyway.

## 3. Preconditions (each recorded in the Training session's banks)

- **Startup gates** on the first seed of EACH arm: R1 compile parity, the K6 freeze, K9(b) from update 1, UpdateFit
  headroom, the update-10 canary. EB's R1 gate and canary run at the bf16 parity rule's derived bars (a canary FATAL is
  INCONCLUSIVE for that seed, never a stop on strength).
- **The first non-empty pool eval** passes and loads pool opponents (Pool % ≠ 0.0). A seed evaluates at 2M and 4M;
  its pool is seeded at the first eval whose bots win rate clears `--self-play-start-wr` 0.55, so a stage-1 seed meets
  this precondition at 4M iff its 2M eval cleared 0.55 (`rb_bf_e_s2002`: Bots 55.9 % at 2M, then Pool 71.0 % at 4M —
  met). Where the 2M eval stays below the gate (`rb_bf_e_s2001` 54.1 %, `rb_bf_eb_s2001` 52.6 %), the 4M eval meets an
  empty pool and its Pool 0.0 % is not a failure: the precondition is **deferred to that seed's stage-2 6M eval**
  (orchestrator, 2026-10-12). Stage 2 reads it on every seed whose first non-empty pool eval falls there.
- **One pin, no fork:** every run's `pin_history` holds P_bf16 only; no run is a fork.
- **K9(b) strict** at its fp32 bars in both arms, with the closing test's amendments 1–2 (§3.7 there) carried over.
- A seed that fails (a run short of its budget, out of restarts after the amendments, a broken precondition) is
  INCONCLUSIVE and replaced by the next seed id (2009, …) of its arm.

## 4. Stage 1, the SCREEN (futility only, never adoption)

- E and EB seeds 2001–2004, each **to 5M steps** (`--steps 5000000`), INTERLEAVED one at a time:
  E2001, EB2001, E2002, EB2002, E2003, EB2003, E2004, EB2004, so drift on the box hits both arms.
- **A 4 × 4 mirrored cross** of the 5M finals: rows EB seeds, columns E seeds; `main.h2h play-many` at P_bf16,
  1,000 mirrored pairs per cell (batches of 500), both sides greedy, schedule seed 0, `--oracle-reveal-mode off`,
  purpose `ab`, family `bf16_strength_steps`, request `bf16_s1_steps`, label `bf16_ab`, the archive's ledger.
- Statistic: `main.h2h.cross.cross_stat` (Δ̂ = mean(h) − 50, V̂ = (s²_R + s²_C)/4, df 6); the two-sided 90 % interval
  Δ̂ ± t_{0.95,6} √V̂, t_{0.95,6} = 1.9432.
- **STOP for futility iff the interval's upper end < −2.0 pp.** Rule 8: an upper end within 1e-9 pp of −2.0 does NOT stop.
- No other stage-1 outcome is read (no early PASS).

## 5. Stage 2, the READ (on no futility stop)

- Seeds 2001–2004 of both arms **continue to 15M**: a `--model <run>/final_model.zip --steps 15000000` resume of
  each run's own 5M final at P_bf16 (same run dir, same pin; the budget changes no schedule at P_bf16: the clip range
  is a constant and the step size is the KL controller's).
- Seeds **2005–2008** of both arms run fresh to 15M (`--steps 15000000`). Both arms do the same in each seed slot.
- Order: interleaved as in stage 1.
- **The 8 × 8 cross** of the 15M finals, played exactly as in §4, request `bf16_s2_steps`.
- **The closing test's rule, unchanged** (`endstate_closing_test_2026-10-09/closing_rule.py` `decide`, on
  `cross_stat`; V̂ = (s²_R + s²_C)/8, df 14, t_{0.95,14} = 1.7613): **PASS iff the 90 % interval's upper end > 0 AND its
  lower end > −2.0 pp**; rule 8 against PASS; INCONCLUSIVE when an input is incomplete or invalid; FAIL otherwise.
- A PASS adopts bf16 into `--arch endstate`'s recipe surface (`recipe.fresh`), with the design_learner_recipe §3.22 row
  in the same commit (PLAN §2d).

## 6. Cost

| part | GPU-h |
|---|---|
| stage 1: 8 seeds × 5M (~1.1 each) + the 4 × 4 cross | ≈ 9 |
| stage 2: E 2001–2004 + 2005–2008 to 15M (~3.5 per full seed) | ≈ 24 |
| stage 2: EB likewise (~3.0 per full seed, window E: −22.6 % update) | ≈ 20 |
| the 8 × 8 cross | ≈ 1 |
| **total** | **≈ 54** (≈ 26 if E could have been reused, §2) |

## 7. Decision record

| date | decision | why |
|---|---|---|
| 2026-10-12 | REGISTERED before any seed, at P_bf16 `f45d0c27`; E re-run (not reused) | PLAN §2d's identity precondition fails at every candidate pin (§2); orchestrator's option (i): E at HEAD doubles as the adopted end state's reference seeds |
| 2026-10-12 | §3's first-non-empty-pool-eval precondition read as unmet by construction in stage 1, checked at every stage-2 seed's 6M eval; the training tree pinned to 0-4,8-12 (§2 CPU split) from `rb_bf_eb_s2001` on | a 5M seed's only pool eval meets an empty pool; `rb_bf_e_s2001` showed contention p90 1.82 with the trainer free to run on the agents' cores (orchestrator's rulings) |
| 2026-10-12 | CORRECTION to the row above: the precondition is not unmet by construction — a seed evaluates at 2M and 4M and is seeded at the first eval clearing `--self-play-start-wr` 0.55, so it is met at 4M when the 2M eval clears the gate (`rb_bf_e_s2002`) and deferred to 6M otherwise (`rb_bf_e_s2001`, `rb_bf_eb_s2001`) | the first bank's reading assumed one pool eval per 5M seed; `rb_bf_e_s2002`'s 4M eval (Pool 71.0 %) refuted it |
