# The bf16 STRENGTH A/B: `--trunk-precision bf16` against fp32, on the end state (REGISTERED 2026-10-12, before any seed)

**Status: REGISTERED 2026-10-12, before any seed; READ 2026-10-14 — stage 1 CONTINUE, stage 2 PASS, and bf16 is ADOPTED into
the production recipe (§7 Results, §8 Decision record).** §§1–6 are the registration, unchanged: they were written before any
cell was read. (Lifted from
[`perf_phase_2026-10-10/PLAN.md`](../research_state/measurements/perf_phase_2026-10-10/PLAN.md) §2d; orchestrator's
call 2026-10-12, option (i). Run by the Training session.)

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
- **The coded rule** (2026-10-12, before any cell): [`futility_rule.py`](../research_state/measurements/bf16_strength_ab_2026-10-12/futility_rule.py)
  `decide`, with its self-check (CONTINUE / STOP / the rule-8 tie on constructed matrices).

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

## 7. Results (2026-10-14)

Both stages were read by the registered rules, unchanged; the cross artifacts, the coded rules' outputs verbatim and the
64 + 16 cells are in
[`measurements/bf16_strength_ab_2026-10-12/`](../research_state/measurements/bf16_strength_ab_2026-10-12/) (`stage1_cross/`,
`stage2_cross/`; the README there carries both `verdict.json` outputs verbatim). Every cross was played at P_bf16
`f45d0c27`, both sides greedy, 1,000 mirrored pairs per cell, schedule seed 0, `--oracle-reveal-mode off`.

| stage | rule | outcome | Δ̂ (pp) | SE (pp) | df | interval (pp) |
|---|---|---|---|---|---|---|
| **1** (4 × 4 at 5M; futility screen) | STOP iff the 90 % upper end < −2.0 (`futility_rule.py`) | **CONTINUE** | −0.122 | 1.765 | 6 | 90 %: [−3.551, +3.307] |
| **2** (8 × 8 at 15M; the READ) | the closing test's `decide`: PASS iff upper > 0 AND lower > −2.0, rule 8 against PASS | **PASS** | **+1.0625** | 1.277 | 14 | 90 %: [−1.186, +3.311] |

- **Stage 2 in full.** `clause_a_includes_zero_or_above` true (upper +3.311 > 0), `clause_b_lower_above_minus_2` true (lower
  −1.186 > −2.0), `near_boundary` none, `t_sup` 0.832, `s2_rows` 8.516, `s2_cols` 4.524, `n_cells_read` 64. Row means (EB
  seeds 2001–2008, pp): 54.69 / 54.28 / 49.86 / 48.84 / 47.80 / 52.70 / 47.44 / 52.88; column means (E seeds): 53.28 / 51.23 /
  52.79 / 48.97 / 47.35 / 50.03 / 53.05 / 51.80.
- **A PASS is NOT-WORSE, not BETTER.** The 90 % interval includes 0 (t_SUP 0.832); the point estimate favours bf16 by 1.06 pp, and
  no superiority claim is made. The bf16 arm's run-to-run spread is the larger (s²_rows 8.52 against 4.52).
- **Speed (REPORTED, never decided).** `train_ms` (TensorBoard, 153 updates per seed, 15M steps): the median of the eight
  seeds' medians is **54.26 s** for E (fp32; per-seed 54.17–54.45) and **42.07 s** for EB (bf16; 41.99–42.15), a ratio of 0.775
  (**−22.5 %**); the pooled means are 55.4 s and 43.0 s (re-read from the runs' event files for the adoption:
  [`speed/train_ms_read.py`](../research_state/measurements/bf16_strength_ab_2026-10-12/speed/train_ms_read.py)).
- **Runs.** 16 runs (eight per arm; seeds 2001–2004 trained in two legs, 5M then a same-pin resume to 15M, so 24 launcher-child
  legs); no `Traceback` in any `launcher_child.full.log` (`speed/train_ms.json` `health`).

## 8. Decision record

| date | decision | why |
|---|---|---|
| 2026-10-12 | REGISTERED before any seed, at P_bf16 `f45d0c27`; E re-run (not reused) | PLAN §2d's identity precondition fails at every candidate pin (§2); orchestrator's option (i): E at HEAD doubles as the adopted end state's reference seeds |
| 2026-10-12 | §3's first-non-empty-pool-eval precondition read as unmet by construction in stage 1, checked at every stage-2 seed's 6M eval; the training tree pinned to 0-4,8-12 (§2 CPU split) from `rb_bf_eb_s2001` on | a 5M seed's only pool eval meets an empty pool; `rb_bf_e_s2001` showed contention p90 1.82 with the trainer free to run on the agents' cores (orchestrator's rulings) |
| 2026-10-12 | CORRECTION to the row above: the precondition is not unmet by construction — a seed evaluates at 2M and 4M and is seeded at the first eval clearing `--self-play-start-wr` 0.55, so it is met at 4M when the 2M eval clears the gate (`rb_bf_e_s2002`) and deferred to 6M otherwise (`rb_bf_e_s2001`, `rb_bf_eb_s2001`) | the first bank's reading assumed one pool eval per 5M seed; `rb_bf_e_s2002`'s 4M eval (Pool 71.0 %) refuted it |
| 2026-10-14 | STAGE 1 read: **CONTINUE** (Δ̂ −0.122 pp, 90 % interval [−3.551, +3.307]; the upper end is not below −2.0) | the coded futility rule (`futility_rule.py`, committed `7201f002` before any cell) on the 4 × 4 cross of the 5M finals | none (a screen only; no early PASS is read) | `stage1_cross/result/verdict.json` |
| 2026-10-14 | STAGE 2 read: **PASS**, and **bf16 is ADOPTED into the production recipe** (`recipe.fresh.trunk_precision` = `bf16`; `--trunk-precision fp32` stays as the explicit opt-out; forks inherit their parent's recorded precision, so `recipe.fork` carries no row) | the closing test's coded rule, unchanged: Δ̂ +1.0625 pp, 90 % interval [−1.186, +3.311], both clauses true (§5: a PASS adopts); `--arch endstate` has been an alias of production since the end state's adoption (2026-10-12), so the recipe the A/B's `--arch endstate` arm ran is the production mirror's | keeping bf16 as an opt-in; defaulting forks of fp32 parents to bf16 (no strength read of a mid-lineage precision switch, and it would confound matched-dose fork comparisons) | `stage2_cross/result/verdict.json`; ledger 2026-10-14 MEASUREMENT (MAJOR); `design_learner_recipe.md` §3.22 row `trunk_precision` + Decision record; `designs/CHANGELOG.md` 2026-10-14 |
