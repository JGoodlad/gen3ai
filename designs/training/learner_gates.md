# The learner's GIGO gates (K9) — the learner golden, behaviour-policy consistency, fail-closed non-finite

M5 Lane K item K9 (`designs/endstate/program_rust_core.md`, the Lane K row and its Decision record).
Three gates on the PPO learner, each aimed at a failure that was INVISIBLE before it: a refactor that
silently changes what an update computes (a), a rollout played by different weights or a different
function than the learner recomputes (b), and a NaN / Inf that silently poisons the weights (c).
Code: `src/agents/training/learner_golden.py`, `src/agents/training/instrumented_ppo/learner_gates.py`.

## (a) The LEARNER GOLDEN (`learner_golden.py`, `learner_golden_test.py`)

**What is pinned.** A production-surface learner REBUILT from a fixed seed (`testkit.fresh_model` —
the production policy kwargs, seed 0, a seeded perturbation so the pointer head is not exactly uniform;
then `main.train.model_build.apply_training_hparams` with `production_args()`, exactly what a launch
applies) runs ONE eager fp32 `InstrumentedMaskablePPO.train()` on a committed buffer:

| input | pinned as |
|---|---|
| the "checkpoint" | NOT committed (3.07M params = 12 MB) — rebuilt from the seeds; its bytes are hashed (`init_params_sha256`) so a construction/seeding change is told apart from an update change |
| the rollout buffer | `learner_golden_buffer.npz` (65 KB): 16 steps x 4 envs = 64 REAL rows of a complete-game rollout on the Rust collector at the production surface — every observation key and label (win targets, belief / item / spread / HP-type labels, intent labels), actions, masks, rewards, episode starts, values, the behaviour log-probs, advantages, returns. Its sha256 is in the golden |
| the recipe | `learner_golden.RECIPE`: the SB3 knobs the production mirror does not carry yet, from `design_learner_recipe.md` §2's live column (clip 0.15, entropy 0.05, lr 2.8e-5, grad clip 0.5, per-micro-batch advantage normalisation), at a scaled-down shape that keeps the live shape's structure: micro 16 x K 3 over 64 rows = one full accumulation group + a RAGGED short group per epoch, 2 epochs |

**What is hashed, and why exact bytes.** `post_params_sha256` = sha256 over every parameter's float32
bytes in `named_parameters()` order (names included); `group_sha256` = the same per parameter group
(first two name components) so a failure says WHERE; `losses` = every logged loss / objective scalar
plus `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`, stored as exact float64 and
compared exactly, so a failure says WHICH term. Diagnostics-only tags (grad-balance shares, ranks, noise
scale, calibration) are NOT compared — a telemetry-only change must not move the golden; any change
that reaches the weights moves the parameter hash anyway. Rounded values were rejected: with 3M values
some always sit on a rounding boundary, so rounding buys no portability and loses teeth.

**Determinism and the torch key.** CPU, eager, fp32, `torch.set_num_threads(1)` for the BUILD and the
update (a CPU reduction order can depend on the thread count), numpy + torch seeded before `train()`.
🚨 **The INIT depends on the thread count too, so `build_learner` pins one thread ITSELF** (F-X5-4, fixed
2026-10-03): SB3's `_build` re-initialises every Linear with `orthogonal_`, a LAPACK QR whose blocked
reduction order follows the BLAS thread count. The RNG draws are identical; the rounding is not (max
|Δ| ~1.1e-6, ~95% of a 512×512 matrix's bytes differ). Built at 8 threads, 15 of the 41 parameter
groups moved (init `ffc668b3…` vs the banked `f476942c…`). The routine test passed only because the
root conftest sets `OMP_NUM_THREADS=1`; a learner built for a re-bake from a CLI shell recorded a
different init. `build_learner` restores the caller's count; `rebuild-buffer` builds its learner at one
thread the same way. 🚨 **The PRODUCTION model build is pinned by the SAME helper** (`gen3_single_thread_init_v1`, F-X5-5, 2026-10-03): `learner_golden._one_thread` IS `utils.torch_state_guard.single_thread_build` (ONE copy; `torch_globals(num_threads=1)`, so the restore is the guard's), and `main.train.model_build.construct_fresh_learner` (the trainer's fresh build), `main.fresh_checkpoint.build_fresh_model` and `rust_rollout.testkit.fresh_model` build inside it — a fresh run's starting weights no longer depend on the core count or `OMP_NUM_THREADS`; the caller's count (the training count) is restored before the first rollout. Measured: +0.03 s on the ~0.95 s production-size build. A resume / fork never needs it: `load_model_snapshot` is STRICT on every key (the ride-along heads included), so the loaded weights overwrite the init. Pinned by `src/main/train/fresh_build_threads_test.py` (the trainer's and the fixture's fresh builds, 1 vs 8 threads in child interpreters, byte-identical `state_dict`; FAILS on a revert) and, for the golden harness, by `learner_golden_threads_test.py` (an 8-thread and a 1-thread build in
child interpreters must equal each other and the banked init; it FAILS on a revert). Two processes
with different `PYTHONHASHSEED` reproduce it bit for bit. **Across torch builds the INIT is identical but
the update is not** (measured 2026-09-30: `gen3ai_torch28` vs `gen3ai_stable` — every pinned loss within
1.2e-7, every parameter group's bytes different), so the golden is KEYED BY `torch.__version__`
(`2.5.1+cu121`, `2.8.0+cu126` today). A build with no entry FAILS; it never skips and never records.

**Cost:** ~3 s (build ~1 s + one update), unmarked (routine tier).

**Teeth (measured 2026-09-30):** changing the advantage-normalisation epsilon `1e-8 → 1e-6`, or the
win-prob critic's coefficient by x1.001, each fails it naming 13+ losses and every parameter group;
reverting passes. `learner_golden_test` keeps one such perturbation (advantage normalisation off) as a
standing test. Every other K9 change below was verified byte-neutral by it under both torch builds.

**Re-recording is EXPLICIT** — the test never writes:

    python -m agents.training.learner_golden record --reason "why the update is MEANT to change"
    python -m agents.training.learner_golden check           # what the test does; prints the diff
    python -m agents.training.learner_golden rebuild-buffer --reason "..."   # obs layout / label schema changed

`record` writes this torch build's entry (keeping the others) with the reason, date, commit and
versions, and appends an append-only `history` row carrying the diff it replaced. Run it under EVERY
interpreter that has an entry. `rebuild-buffer` (needs the `rust_env` self-check build) records a new
buffer and invalidates every entry (they were computed on different input).

**Scope limits.** Not pinned: the rollout and GAE (the buffer's advantages / returns are inputs), the
KL→LR controller and every other callback (outside `train()`), CUDA / compiled numerics (the
compile parity gate and K6's canary own those). When K10(a)'s recipe block lands, `RECIPE` should read
it and the golden be re-recorded.

### The SECOND entry: `--belief-tokens fixed_mass` (X5 U6, 2026-10-04)

`learner_golden.ARMS` declares the non-default arms. Each is recorded under `arms.<name>` in `learner_golden.json`,
beside the blob entries, which a non-default record never touches. `fixed_mass` = `production_args()` +
`belief_tokens = fixed_mass`. It differs from blob in three deliberate ways, each forced:

| | blob (top level) | `arms.fixed_mass` | why |
|---|---|---|---|
| perturbation | order-keyed (`parity_probe._noise`) | NAME-KEYED (`_keyed_noise`: one generator per parameter, seeded by `sha256(seed:name)`) | the arm adds X5 groups and retires α / β / `belief_slots`; an order-keyed draw would move every later parameter's noise, so no shared group could be compared with blob's |
| buffer | `learner_golden_buffer.npz` | `learner_golden_buffer_fixed_mass.npz`, the same rollout recipe at run seed 18 | the observation spaces are IDENTICAL, but the blob buffer's behaviour log-probs are the blob learner's (K9(b) of the fixed_mass learner on it: max \|Δ\| 0.51). Seed 17 gave no OTHER_move-dead row (design §6.4: ≥ 2 rows per X5 case) |
| recorded extras | — | `init_group_sha256`, `fp64_reference`, `behaviour` (the K9(b) read), `coverage` | design §6: what proves a re-bake correct, not merely new |

    python -m agents.training.learner_golden check  --arm fixed_mass
    python -m agents.training.learner_golden record --arm fixed_mass --reason "..."
    python -m agents.training.learner_golden rebuild-buffer --arm fixed_mass --reason "..."

**The fp64 reference** (`learner_golden_fp64.py`).
- **What runs.** At the seeded INIT, on the arm's buffer with the intent labels aligned as `train()` aligns them:
  ONE `micro_step` (region R1, every term of fold steps 1–3a), at fp32 and on a float64 copy under `Fp64Mode`. The
  mode promotes every float32 operand of every torch op, and `.float()` means `.double()`.
- **Rule 8.** Rows whose K9(b) tie margin is under `FP32_TIE_EPS` are excluded from both runs: 2 of 64.
- **What is compared.** Every term within `5e-6 + 1e-5·|t64|`; seven key gradients (δ_θ, OTHER's map, the hypothesis
  marker, the flat pointer, the set-BCE head, the policy, the critic) within relative L2 `5e-5`.
- **Measured (recorded).** Terms ≤ 3.3e-7 abs and ≤ 1.3e-6 relative; gradients ≤ 1.5e-6.
- **Exactness.** Both precisions reproduce EXACTLY within one torch build and one thread.
- **Teeth.** An fp32-only 1e-3 logit shift fails it.
- **Limit.** It is the same code at two precisions. The independent fp64 numpy checks of the construction
  (Σπ = k, a direct bisection, OTHER's −1e9 iff its tail is empty) are tests of their own.

**What `learner_golden_fixed_mass_test.py` pins (16 tests, ~26 s).**
- **Reproduction.** The update reproduces exactly with K9(b) ON. The probe passes: max |Δ| 4.8e-7, excluded 6.25 %
  (1 of 16) against the 0.15 ceiling. It needed the X5 fix F-X5-44: before it, 47 % of rows were excluded.
- **Not vacuous.** The X5 loss keys are logged and nonzero, α / β's are absent, every X5 group moved.
- **Shared init.** Every shared group's INITIAL bytes equal blob's built with the same keyed noise.
- **Coverage.** OTHER_species live 36 / dead 28, OTHER_move live 60 / dead 4.
- **Teeth.** Four planted X5 perturbations each FAIL it, moving 14–15 losses and 35 groups, both X5 groups among them,
  never the INIT:
  - τ + 1e-2 in the construction;
  - the log-π key bias's sign;
  - OTHER's column read at the next slot;
  - OTHER_move masked out of the flat pointer.

  All four together leave the blob golden byte-identical. A blob-only plant (α's head) fails blob and leaves
  fixed_mass byte-identical.
- **δ_θ isolation.** With every set BCE's logits detached, δ_θ's bytes do not move in the real update.
- **B ride-along.** The arm + B (`--ridealong-opp 2`) equals the entry in every recorded group and loss.

`learner_golden_threads_test` adds two processes at `PYTHONHASHSEED` 0 / 4242 and 1 / 8 BLAS threads: identical
bytes, equal to the recorded entry.

**Scope note.** A blob buffer rebuild replays the same games but moves the behaviour columns by ≤ 7.2e-7 (T2
rounding, F-X5-45), so re-record after any rebuild. The blob entry predates `init_group_sha256` (F-X5-46). `diff`
compares the field only where it is recorded, and the next blob re-record adds it.

## (b) Behaviour-policy consistency (`--behaviour-check`, default `fatal`)

Before any optimizer step of every update, the learner's recomputed log π(a|s) must equal the stored
behaviour log-prob: max |Δ| < 1e-4, else `BehaviourMismatch` (`rust_rollout/consistency.py` — one type,
one bar). ONE implementation runs it (`learner_gates.behaviour_gate_mode` says only whether): Lane G's
pre-loop probe — its own forward on one micro-batch while the buffer is still `[n_steps, n_envs]`, so
current rows are held to the bar and older rows are age-bucketed into `staleness/*` (`rust_collector.md`);
a buffer with no version record (a test's toy learner, the startup dry update's fixture) is judged as every
row current. **The python core's IN-LOOP variant is DELETED** (deletion pass U4, 2026-10-02): it compared the
first micro-batch's own forward in the loop for a buffer with no versions, which only the Python env core
produced; with that core gone, keeping it meant a second K9(b) implementation that only tests reached.
Its tests now judge through the probe and the shared enforcement (`learner_gates_test`).

It logs `behaviour/max_abs_dlogp_current`, `behaviour/p99_abs_dlogp_current`, `rows_current`,
`rows_probed`, one `behaviour/bar_<statistic>` per condition and
`behaviour/excluded_frac`, `rows_excluded`, `rows_judged`, `max_abs_dlogp_judged`,
`max_abs_dlogp_excluded` and `tie_eps` (and the Rust probe `probe_forward_ms`).

**The gate is ONE table, at fp32 matmul precision `highest` — the only precision** (`consistency.BEHAVIOUR_GATE`,
read by Lane G's probe through `judge_behaviour`; TF32 was retired, deletion pass
K2, and a process at any other precision is refused by `consistency.behaviour_gate()` with a typed
`UndeclaredPrecision`). Every condition must hold on the micro-batch's current rows:

| condition | catches | bar | FATAL when |
|---|---|---|---|
| max \|Δ\| over the rows NOT at a selection tie, + the excluded share | every fault class | 1e-4; excluded share < 0.15 | the FIRST violation — **DETERMINISTIC** (below): a row within a relative margin `FP32_TIE_EPS` = 2e-4 of a discrete cutoff is excluded, every other row is judged |

**Every violation** prints `🚨` (or `🛑` when fatal) with the condition, its value and its bar, logs the
offending rows (index, |Δ|, action, mask — the largest 10; the p99 for context), and appends a JSON line
with the largest 200 to `<run_dir>/behaviour_violations.jsonl` (`consistency.VIOLATION_DUMP`) BEFORE it
raises. There is no persistence: the first violation is FATAL under `--behaviour-check fatal`; a NaN |Δ| is
FATAL at once (never rounding); `--behaviour-check warn` only ever warns. `behaviour/violations_total_<statistic>`
counts them on the dashboard every update (`warn` mode's tally).

**On the Rust path a violation's dump is enough to root-cause the row offline**
(`gen3_behaviour_provenance_v1`, after sizing arm A2's single-row FATAL, 2026-10-01: one row of 1,024 at
|Δ| 0.0389 and no log-probs in the dump to tell numerics from a fault). Each dumped row carries:
- the stored log μ and the recomputed log π;
- BOTH full masked distributions (the stored one is the row the keyed draw used, carried from the arena
  by `store.row_provenance`; the recomputed one is the probe forward's own `_last_pi_distribution`),
  each with its entropy, top-1 minus top-2 margin, argmax and p(action);
- the row's collection provenance: env, episode, decision index, policy version, the T2 slot that
  served it (`ROW_SOURCE_FORK` for a fork-arm branch row) and the keyed draw (`u`, `draw_margin`).

The record also carries:
- the `route`: T2's backend, buckets, lanes and current slot, plus the keyed run seed (with
  env / episode / dec_n the row is replayable on the deterministic core);
- a FULL-BUFFER `scan` (`consistency.scan_current`): every current row through the same forward, so one
  row is told apart from many, with the worst 32 in full.

Two artifacts are written beside it:
- `behaviour_violation_u<n>_rows.npz`: the violating rows' observations;
- `behaviour_violation_u<n>_policy.pt`: the weights that played them, for the first 3 violations of a
  process.

A clean update pays nothing.

**The fp32 rule is DETERMINISTIC (`gen3_behaviour_tie_exclusion_v1`, owner 2026-10-01; every number
from [`measurements/k9_behaviour_exclusion/result.json`](../research_state/measurements/k9_behaviour_exclusion/result.json)).**
The owner: *"toss out ones where the cutoff would be sensitive to a rounding error … and then
deterministically pass or fail"*. It **supersedes the probabilistic TIE RULE** (`gen3_behaviour_tie_rule_v1`,
same day: a violation warned and was FATAL on 4 consecutive updates, on a count over the healthy rate, or
at no tie — numbers in [`k9_behaviour_tail/result.json`](../research_state/measurements/k9_behaviour_tail/result.json)).

*Why a row can jump.* Sizing arm A2 died on ONE row of 1,024 at |Δ| 0.0389 (p99 2.6e-6). The tail sweep
(`k9_behaviour_tail/`) found the mechanism: the forward is PIECEWISE-DISCONTINUOUS — it SELECTS (the
threat-seat `topk` and its `>=` cutoff in `pointer_head.py`, the damage op's candidate `topk`, the
dominant-move `argmax` in `damage_op.py`) and THRESHOLDS — and where one sits within a rounding error of
its cutoff, T2 (compiled, at its bucket) and the learner (eager, at its batch) resolve it differently.
T2 reproduced its own stored value bit-for-bit; a few-ulp weight jitter of the eager forward landed on
it; the flipped gaps were 5e-8 to 1.7e-7 in fp64 (`TECH_DEBT_BACKLOG.md` §2(b)).

*The rule.*
1. **Every discrete op of the forward is DECLARED** in `agents/model/selection_sites.py` — each selection
   call, value-position comparison and float → int cast of a forward module, keyed by source: a MARGIN
   rule for a score (19 sites: the five candidate `topk`s, the `>=` top-K cutoff, the two dominant-move
   `argmax`es — GATED, so a slot whose max is at or below the gate's 1e-6 is masked and cannot tie — their
   two gates, and the other score / weight-free-arithmetic thresholds), an EXACT reason for the rest
   (observation reads, constant tables, integers, values gathered at a declared selection's index, the
   move-ID match). `selection_sites_test` fails on an undeclared or stale entry (an AST scan of every
   forward module), on a line mixing the two classes, and on an EXACT site whose operands move under a
   few-ulp weight jitter; at run time an undeclared op on a float operand is a typed `TieMarginError`.
2. **Each judged row's TIE MARGIN** is computed from the probe's OWN forward — on the Rust core the
   probe forward itself runs under `rust_rollout/tie_margins.TieMargins` (a `TorchFunctionMode` that
   reads each declared op's operands after the op ran; the forward is unchanged, and training forwards
   never run under it); on the Python core one no-grad eager forward of the first micro-batch runs under
   it BEFORE R1's own forward (so every stash the fold reads is R1's; no optimizer step has run). The
   margin is RELATIVE (|a − b| / max(|a|, |b|): fp32 rounding scales with the values compared) — a
   `topk`'s k-th vs (k+1)-th, an `argmax`'s top-1 vs top-2, a threshold's distance — the minimum over
   every MARGIN site. An EXACT tie is 0.
3. **A row whose margin is below `FP32_TIE_EPS` = 2e-4 is EXCLUDED**; exact ties always.
4. **Every other current row is JUDGED: any |Δ| ≥ 1e-4 is FATAL on the first update.** No persistence,
   no count. A judged row resolves every selection identically in both forwards, so its |Δ| is
   continuous fp32 noise.
5. **The excluded share must stay under `FP32_EXCLUDED_CEILING` = 0.15** (FATAL otherwise): a fault that
   pushed many rows onto ties would otherwise hide from the judgement.

*The measurement* (`k9_behaviour_exclusion/`, `sweep.py` + `derive.py`, criteria declared before the
numbers were read): the tail sweep's setup re-run with margins — A2's 4.0M checkpoint and its 2
snapshots, the production rust path (T2 graph backend, buckets 8 / 48, 7 lanes, N = 48), nothing
training, every one of 3,538,944 rows of 36 fills through the probe forward; torch 2.8, RTX 3080 Ti.

| quantity | value | derivation |
|---|---|---|
| the ROUNDING SCALE R of the margins | **1.44e-5** | the largest \|m_learner − m_variant\| over 73,728 rows × every declared site, the variants being T2-like forwards: eval / no-grad at buckets 48 (≤ 3.6e-6) and 8 (≤ 4.8e-6), a few-ulp weight jitter (1.44e-5 — the largest; it is what reproduced T2's stored values in the tail sweep) |
| `FP32_TIE_EPS` | **2e-4** | 10 × R, rounded up to 1-2-5 |
| rows over the bar (all rows) | 4, at margins ≤ **4.2e-7** | every real flip sat ~480× under epsilon (and under 1e-6) |
| JUDGED rows over the bar | **0 of 3,407,893** | judged max \|Δ\| 2.1e-5 — 4.7× under 1e-4; by margin decade the max is 2.1e-5 at [1e-3, 1e-2) and ≤ 1.9e-5 above |
| excluded share | **3.70 %** (largest 1,024-row block 6.25 %) | 2.0 % dominant-move `argmax` exact ties with the gate open, 0.6 % fixed damage == current HP exactly, the rest near-ties of the candidate `topk` / its cutoff |
| `FP32_EXCLUDED_CEILING` | **0.15** | the smallest of 0.02 / 0.05 / 0.10 / 0.15 / … at ≥ 3 × the pooled share and ≥ 1.5 × the largest block; fresh, 4.0M and 75M (N0 final) weights read 4.2 / 3.9 / 4.2 % at epsilon (CPU, 4,096 of A2's real rows) |
| planted faults (the real probe, at epsilon) | **all FATAL on the FIRST update** | one-step-stale weights (an Adam-sized 2.8e-5 step on every weight); ONE wrong-action row at margin 0.128; one env's obs / mask column shifted by one decision. The unmodified buffer passes, before and after |
| cost | **+27.7 ms** on a 62.6 ms probe forward of 2,048 rows | ~0.07 % of an update (~40 s) |

*The teeth, stated.* A fault confined to ONE row is missed only if that row is excluded: P ≈ 0.037 at
the healthy share. A fault that moves many rows is caught by its judged rows, or by the ceiling.
`consistency_test` re-derives epsilon and the ceiling from `result.json` and fails on a changed constant.

*Exact ties are excluded, per the owner's rule.* Most are flip-proof in practice — two candidates with
bit-identical scores (a capped damage roll on a saturated / revealed belief weight; fixed damage equal to
the remaining HP) resolved by the same lowest-index tie-break in eager and Inductor — but the rule does
not try to prove that. A refinement that counts an argmax tie only when the tied candidates' gathered
payloads differ would return ~2 points of teeth (not built).

*UNVERIFIED:* the rounding scale is measured against EAGER T2-like variants and a weight jitter, not
against T2's own compiled intermediate values (a `TorchFunctionMode` cannot see inside a compiled
graph); the sweep's 4 real flips — all ~480× under epsilon — are the end-to-end check against the real
T2. The excluded share is measured on A2's states; a live run's ecology may differ (the ceiling FATAL
names the re-measurement).

**The measurement** — `designs/research_state/measurements/k9_behaviour_bar_2026-09-30/`: `result.json`
(`measure.py`, `derive.py`: the healthy distribution and the stale faults) and `corrupt_result.json`
(`corrupt.py`: the localized faults). RTX 3080 Ti, torch 2.5.1. Rows: per seed a real complete-game
Rust-collector rollout at the production surface (48 envs x 128 steps = 6,144 rows); the learner is the
learner golden's production learner with a per-seed perturbation; 4 seeds x 3 REAL updates each, eager
and `--compile-trainer`. The rollout side is the python core's (eval mode, no grad, 48-row batches), the
learner side a 2,048-row micro-batch (train mode, grad). n = 147,456 rows, 72 micro-batches (stale faults
72 / 48; localized faults: 4 seeds x 2 states x 3 micro-batches, eager). (The same files carry a TF32
column of that first measurement; TF32 was retired and no gate reads it.)

**THE TABLE** — the statistic the gate reads over a 2,048-row micro-batch (per-row figures where noted),
healthy and under each fault class; margin = the fault's SMALLEST micro-batch value / the bar,
headroom = the bar / the healthy LARGEST. fp32 is CUDA `highest` (healthy, stale) or CPU fp32 (localized:
the fault SIGNAL — the CUDA fp32 localized units were not run).

| | fp32: max (bar 1e-4) |
|---|---|
| healthy per-row p99.9 / max (n = 147,456 rows) | 9.5e-7 / 1.9e-6 |
| **healthy micro-batch, largest of 72** | 1.9e-6 — **headroom 52x** |
| healthy, eval/train MODE alone (per-row max) | 1.8e-6 |
| GLOBAL: rollout ONE optimizer step stale, lr 2.8e-5 (72) | 0.020 — **200x** |
| GLOBAL: one optimizer step stale, lr 3e-4 (72) | 0.16 — 1,600x |
| GLOBAL: one full update stale (48) | 0.108 — 1,085x |
| GLOBAL: eval/train-mode mismatch | VACUOUS (below) |
| LOCALIZED, 0.5 % of rows: wrong action index (24) | 0.244 — 2,440x |
| LOCALIZED, 0.5 %: obs swap / rows misaligned (24) | 0.549 — 5,490x |
| LOCALIZED, 0.5 %: mask mismatch (24) | 0.457 — 4,570x |

Every GLOBAL and every LOCALIZED fault micro-batch is caught by the gate; no healthy micro-batch trips
it. These are a FRESH (near-uniform) learner's signals; a trained, sharper policy should separate wrong
actions more (**UNVERIFIED** on a trained checkpoint). ⚠️ An earlier pass (1,024-row micro-batches, the
same design) saw one fp32 healthy row at 2.6e-5 — 3.9x under 1e-4, the thinnest fp32 headroom observed.

**The eval/train-MODE fault is VACUOUS for this policy.** It has no dropout and no batch norm (checked
on the built module tree), so eval and train mode compute the same FUNCTION; the only mode-dependent
code is kernel choice (`nn.TransformerEncoderLayer`'s eval fast path, gradient checkpointing, the
`is_grad_enabled` branches that stash training targets). That component is measured above ("mode
ALONE") and is inside the healthy distribution the bar is set from. No real mode fault can be
constructed; a future dropout / batch-norm layer would be one, and this gate is what would see it.

**Measured at fp32 (CPU), on the deleted python-core path:** the `--debug` smoke (2,048-row micro-batches): |Δ| = 0
on update 1 (a fresh pointer head is uniform) and 2.38e-7 after; the learner golden's buffer (Rust T2
eager log-probs vs the learner): 3.6e-7. Lane G's GPU read on the Rust path (compiled learner vs T2
graph): 1.45e-5.

The four paused learner-battery argvs (L95, T32_STANDIN, Cfix, T32b_sentinel — none typing
`--behaviour-check`) parse (with the flags deleted since they were recorded stripped, `--matmul-precision`
among them) and resolve to `fatal` (`learner_gates_test`).

**The other legitimate reasons it could differ, checked:** batch composition — the rollout forwards
`n_envs` rows, the learner a micro-batch (the healthy rows above); `--compile-trainer` — measured in
both arms above; grad-checkpointing — the recompute is bit-exact
(dropout 0, `use_reentrant=False`); fork-arm branch rows — their log-probs come from the parent's live
policy (the Python arm's `fork_buffer.build_branch_rows`, deleted in L5; the Rust pass re-serves row 0 from the current trainee slot, forks.md §14.5), i.e. the behaviour policy; `_align_opp_intent_labels` — rewrites label keys only, never a policy input.
**UNVERIFIED:** a real CUDA python-core launch (no GPU training during the M5 halt) — the CUDA numbers
are the production learner's forwards on real rollout rows, not a live rollout loop.

## (c) Every non-finite loss / gradient path is FAIL-CLOSED (`NonFiniteLearnerError`)

Before K9(c) nothing in `train()` looked: a NaN loss back-propagated NaN gradients, `clip_grad_norm_`
(`error_if_nonfinite=False`) scaled every gradient by `max_norm / NaN` in place, and `optimizer.step()`
wrote NaN into every parameter and into Adam's moments — silently. The gates, all BEFORE the optimizer
can move anything (so a crash-save holds the last finite weights):

| gate | where | catches |
|---|---|---|
| `check_buffer_finite` | once per update, right after the intent-label alignment — before any forward (it once also ran BEFORE PopArt's advance, which rewrote `value_net` outside the optimizer; PopArt is deleted) | a NaN/Inf reward, value, behaviour log-prob, advantage, return, or FLOAT label key (the flat `observation` is not scanned: an input whose NaN reaches the loss, and a full scan is ~0.3 s at production size) |
| `check_loss_finite` | once per micro-batch, on the assembled loss, before the grad-balance / noise probes and the backward. Since K8 the region R1's `isfinite(loss)` rides the micro-batch's ONE host read (`micro_step.pack`); the full check (its own read) runs when the eager tail folded a term onto R1's loss, or to NAME the term(s) on a failure — same timing, same message | any term; NAMES the non-finite term(s) (policy, entropy, value, and every `aux_probe_terms` entry) |
| `check_kl_finite` | per micro-batch, on sb3's own approx-KL host read | an Inf KL under a FINITE loss (an overflowed ratio on a positive-advantage row takes the clipped branch) |
| `clip_grad_norm_checked` | every optimizer step (the in-loop step and the accumulation flush) | a NaN/Inf gradient from a finite loss; `error_if_nonfinite=True` raises BEFORE the in-place scaling, so the named parameters are the ones the backward poisoned |

### The audit (2026-09-30) — every site, its verdict, and what was done

**Absorbed BEFORE the total (a total-loss check could not see these) — FIXED:**

| site | was | fix |
|---|---|---|
| `ppo.py` β reachability `_reach = isfinite(target logit)` | dropped a NaN target logit's row like a deliberate −inf one (weighted path: the row left the graph) | `~isneginf(...)`: only −inf is unreachable; a NaN stays supervised and reaches the loss check. Byte-identical otherwise (golden) |
| `opp_intent.set_valued_switch_loss` `avail = believed & isfinite` | a row whose only believed slots were NaN was dropped | `~isneginf`, same reasoning |
| `ppo.py` advantage normalisation | a ONE-row final micro-batch (python core, n_steps·n_envs ≡ 1 mod the micro-batch; the Rust collector's target is a multiple of it) had std() = NaN → NaN into every weight, silently | stock SB3 PPO's `numel() > 1` guard (sb3_contrib's MaskablePPO lacks it) |
| `rust_rollout/collector.py` `won = reward > 0` | a NaN reward became a finite LOSS label | the NaN reward is refused at the buffer check |
| `adaptive_lr_callback.py` (both controllers) | a NaN KL froze `_kl_ema` (and the LR) for the rest of the run; an Inf walked the LR to `min_lr` | typed FATAL (the train-side `check_kl_finite` is the first line) |

**Reach the total — covered by the loss / gradient checks:** the clipped surrogate, the value MSE (plain /
clipped), entropy; the win-prob BCE and dense aux (multiplicative masks: NaN x 0 = NaN); the
the cf binomial / beta-binomial / shadow losses; the belief bank
(index selection: a selected NaN propagates); the switch-branch / intent-conditional weights
(`clamp` keeps NaN); SB3's and the Rust collector's GAE (no checks of their own — the buffer check
covers their output); the noise-scale EMA (reads the same `.grad`); a NaN INPUT row (torch's `Categorical` argument
validation raises before any loss — untyped, pinned by `learner_gates_test`).

**LEGIT masks:** `damage_op_blocks.py` `where(isfinite(cheapest))` (every candidate is a constant or
the +inf "no cure path" sentinel, no learned value); sb3_contrib's `where(mask, logits, -1e8)` (illegal
slots, zero gradient); `fork_arm.py` gap `isfinite` (selection of which states fork); integer label keys (int64 —
a NaN cannot be stored).

**DETACHED METERS that may fail open by design — the ride-along heads** (`instrumented_ppo/ridealong_terms.py`,
`gen3_ridealong_heads_v1`, OFF in production): a non-finite ride-along loss or gradient is never stepped
and DISABLES the heads for the rest of the process — LOUDLY (a `🛑 [RIDE-ALONG]` line with the grad norm
and every loss) and on the dashboard (`ridealong/disabled` = 1, recorded every update) — never the run.
Legitimate, verified: the heads' inputs are `.detach()`ed stashes, they have their OWN optimizer (no
PPO param group) and their gradients are set back to `None` before PPO's loss exists, so they never
reach a PPO gradient, the policy, V or the LR (`ridealong_heads_test`: no ride-along loss reaches a
trunk / policy / V parameter; `ridealong_update_test`: one update ON vs OFF is bit-identical in params,
PPO optimizer state, PPO scalars and RNG; no callback or controller reads a `ridealong/*` tag). K9(c)'s
own checks are unaffected: the heads' cleared gradients are invisible to `clip_grad_norm_checked`, and
a NaN in the stashes they read is also in PPO's loss, where `check_loss_finite` fires.

**TELEMETRY only (never a gradient / weight / LR):** `calibration.py`, `signal_metrics.py`,
`scaffolding.py`, `value_terms._masked_auc`, the NaN-omitting metric folds in `ppo.py`, the fork-arm
rate functions, the `try/except` telemetry blocks in
`noise_scale*`, `rollout_probes`, `capacity_terms`; `consistency.py`'s own bar (`not worst
< BAR` also fails on NaN).

**OPEN — off in production, recorded rather than changed:** `keyed_draw.py` reports a NaN logit row as "no legal action" (fail-closed, misleading message);
`--target-kl nan` parses (no finiteness validation in the parser).

**The exit side (cutover-prep, `743008c1`).** Every K9(c) raise is `main.exit_codes.NonFiniteLearnerError`
(a `FloatingPointError`; `learner_gates.nonfinite(msg)` builds it) with its message tagged
`[Learner] FATAL`. The trainer's fail-fast handlers map it through `exit_code_for` to
`TrainExitCode.FATAL_NONFINITE` (4) and the launcher STOPS on 4 instead of resuming the checkpoint that
produced it — a restart would replay the same update. K9 defines no class of its own.

## K6 — the CUDA memory TREND: a leak detector for a clean early stop, NOT a gate (`cuda_memory_trend.py`)

The declared lifecycle's MEMORY half (`designs/endstate/program_rust_core.md`, the M5 DESIGN PRINCIPLE
and its Decision record), as the orchestrator redefined it: a slow leak of live CUDA tensors must end
in a deliberate stop WITH a checkpoint, not in an out-of-memory mid-update and a restart that replays
the leak. Fragmentation and one-off step-ups are expected and never stop a run. **Status: BUILT and
WIRED** (`learner_lifecycle.CudaMemoryWatch`, attached with the freeze guard — see
[`learner_lifecycle.md`](learner_lifecycle.md) "The memory half": a sample after every rollout and
every update, the projection logged + recorded at every window close, a STOP raising
`CudaMemoryLeakError` — `final_model_exception.zip` saved, exit `FATAL_CUDA_LEAK` (6), restarted by the launcher from that checkpoint at most twice per session).

**What it reads.** Once per update `sample_cuda(device, update=, phase=)` takes
`torch.cuda.memory_stats` (allocated, reserved, active, inactive-split, segments, `num_alloc_retries`,
`num_ooms`, peaks since the last sample) and `mem_get_info` (device free/total; other processes share
the card).

**What it fits, and why.** `MemoryTrend` is pure, with no torch.

- The leak signal is the window FLOOR: the min of the QUIESCENT allocated bytes over
  `WINDOW_UPDATES = 2` updates. Fragmentation never moves it; a leak moves it every update.
- Reserved moves in permanent steps, so reserved is never fit. It is the DEMAND (peak reserved) and,
  with the device's free bytes, the CEILING (reserved + free − 512 MiB).

**The verdicts.**

- **SUSTAINED:** over the last 9 windows split into 3 odd groups, each group median exceeds the last
  by more than `NOISE_BAR_BYTES` = 8 MiB, AND the Theil-Sen slope is ≥ 1 MiB/update. A single step
  can raise only one median.
- **STOP:** SUSTAINED and the projected OOM is inside `HORIZON_UPDATES = 25` (the default periodic
  checkpoint cadence, or the integrator's own) on `PERSIST_WINDOWS = 3` consecutive closes.
- **WARN, never STOP:** a SUSTAINED read beyond the horizon (carries its projection); a floor step
  > 64 MiB; a reserved step > 256 MiB; a new `num_alloc_retries` / `num_ooms`; headroom under the
  margin.
- The first 2 updates of a process are never windowed (Adam's lazily created state).
- TB: `lifecycle/cuda_*` (level, floor, demand, ceiling, slope, updates-to-ceiling, segments after
  freeze, retries).

**Calibration (measured 2026-09-30, torch 2.8, RTX 3080 Ti, production-shape learner on a real
98,304-row buffer, 5 processes x 15–57 updates, one at the production 10 epochs; `designs/research_state/measurements/k6_k8/memory/`):**

- The healthy allocated floor holds within **2.0 MiB**, so the 8 MiB bar has 4x headroom.
- Reserved settles by the first diagnostics update: +16 MiB after the warm-up under the default
  allocator, then flat for 47 updates.
- Peak demand is 5.6 GiB against a 12 GiB card (diagnostics updates +1.2 GiB over plain ones).
- 0 alloc retries, 0 OOMs, every replayed window OK.

**False trips:**

- **Impossible by construction** while the healthy floor band stays under the 8 MiB bar, because
  every group-median rise is bounded by the band.
- In 60-window simulated processes on a card with 1 GiB headroom (2,000 per magnification), the
  measured residuals magnified ×8 produce no SUSTAINED read (×16: 0.05%), and even ×64 produces no
  STOP.

**Lead:**

- A slow leak reads SUSTAINED about 12 updates after it starts.
- It STOPs `25 + 512 MiB / slope` updates before the OOM: 29 updates at 60 MiB/update, 83 at
  8 MiB/update.
- ⚠️ A FAST leak (≳ 300 MiB/update here, an OOM within about 12–20 updates of onset) crashes before
  the span fills. That case is the launcher's restart policy, not this detector's.
- A leak under about 1.3 MiB/update reads OK but is visible in `lifecycle/cuda_floor_mib`.

**`expandable_segments:True` is not recommended** (same README):

- **Reserved:** −60 MiB (1%), but +156 MiB of growth after the warm-up.
- **Segment counters:** `memory_stats` then reports ZERO segments, which blinds every
  `cuda_segments_after_freeze` counter, T2's guard included. T2's graph backend still starts and
  passes its parity gate under it, so the setting is compatible but blind. ⚠️ The launcher sets it for
  EVERY child (`main/launcher/child.py`), and it hid a real climb: sizing arm B's per-update side stream
  stranded +1.86 GiB of reserved with every segment count at 0. K6's own counters now take a segment
  CENSUS from `memory_snapshot()` and read `reserved_after_freeze` / `streams_after_freeze`
  (`gen3_reserved_after_freeze_v1`, `learner_lifecycle.md` "The staged batch's stream"); T2's per-flush
  guard is still blind (a TECH-DEBT row).
- **Update time:** not separable from box contention (load1 7–19; one unit's plain update ranged
  9.9–20.9 s).
- **Fragmentation** under the default allocator is already a plateau.

**ASSUMED, UNVERIFIED:**

- a live run's floor wobble and callbacks behave like this harness, which ran no callbacks and no T2;
- on the Rust core (the only core), T2's declared slots and the in-process eval cycle share the trainer's card
  and are seen only through the ceiling.
