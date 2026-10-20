# Adversarial review of FP32_ALT (`d4af9f1e`, `gen3_tau_fp32_alt_v1`), 2026-10-20

**Scope.** The split-residual τ solve that replaces T62's float64 one: the proof
(`designs/endstate/design_x5_belief_tokens.md` §3.2), the code (`src/agents/model/hypothesis_set.py`
`fixed_size_tau`), the tests (`src/agents/model/tau_solve_test.py`), and the evidence (this directory's README).
The failure it must prevent is X26's τ class (`../x26_a3_stop_316M_2026-10-18/`): on a k = 1 row with one
near-certain candidate, the naive fp32 sum fixes τ only to ulp(1) / slope ≈ 1e-2. The compiled rollout and the
eager learner then land at different points of that band, and K9(b) reads a 1e-3 gap.

**Method.** I read the code and docs, then ran CPU checks (`review/`, below): an independent float64 reference,
worst-case summation orders, edge cases, underflow, Inductor CPU compiled against eager, and gradients. I also
reverted each claim in a scratch copy and confirmed that its new test fails.

## Verdict: **SOUND-WITH-FIXES**

The mathematics is right. The code implements it faithfully. The X26 τ class is closed: the error is at the ulp
level, the evidence was measured on X26's own violators, and the tests have teeth.

None of the fixes below changes the solve. They fall into three groups:
- an overstated constant in the proof;
- one unmeasured program pair (Triton on CUDA);
- a live run, outside FP32_ALT's code, that runs T62's float64 solve.

FP32_ALT is also **more accurate than the float64 solve it replaces** past a ~60-nat lead. On random rows with an
80-nat lead, fp64's naive sum missed τ by up to 2.3 nats, while FP32_ALT stayed within 4.0e-6.

## The proof, checked line by line

| step | verdict |
|---|---|
| Σσ(x) > k ⟺ Σh(x) > k − \|H\|, with h = σ(x) for x ≤ 0 and −σ(−x) for x > 0 | **Correct.** It is an identity, since σ(x) = 1 − σ(−x). The code derives H and the sign from ONE predicate tensor (`in_h = x > 0`), so a tie at x = 0 is consistent by construction: h(0) = ½ and 0 ∉ H. |
| k − \|H\| is exact | **Correct.** `k_safe - in_h.sum(-1)` is int64, then cast once; \|k − \|H\|\| ≤ 400 is exact in fp32. The Inductor C++ kernel keeps it int64 (`tmp_acc1`, `c10::convert<float>(tmp8 - tmp9)`). |
| S' = Σπ(1 − π) ≥ ½Σmin(π, 1 − π) = ½(R + D) | **Correct**, because max(π, 1 − π) ≥ ½. |
| log(R + D) is Lipschitz-1 in τ | **Correct**, because \|d σ(−\|x\|)/dτ\| = σ(1 − σ) ≤ σ(−\|x\|). |
| Direction right whenever \|τ − τ\*\| > −ln(1 − 2c) | **Correct.** Integrating S' from τ\* gives \|S − k\| ≥ ½(R + D)(τ)(1 − e^{−Δ}), which exceeds c(R + D)(τ) exactly then. Wrong decisions inside the 2c band leave the bisection within 2c + 2^-64·width of τ\*. |
| "c ≈ (depth + 2)u, depth ≤ 9 for a 400-wide tree" | **Overstated (F2).** The bound is valid for any order, but the constant applies only to a tree reduction. The CPU Inductor program is a single sequential chain of length n. |
| Underflow past "~165 nats" (design doc, README) / "~175 nats" (docstring) | **The design doc is right; the docstring is wrong.** Measured within 2.5e-6 up to a 165-nat lead, then 0.28 at 170, 2.0 at 175 and 13 at 200. It is unreachable (F5). |
| "τ needs no gradient", ∂BCE/∂τ = Σ(π − y) = 0 | **Correct and unchanged.** The solve is under `no_grad` in every branch. `set_bce` scores only rows whose labels count exactly k. The BCE gradient under FP32_ALT and under fp64 differs by 1.9e-9 absolute (4.8e-7 of the largest gradient) on 256 mixed rows (`review/check_compile.log`). |

## Findings, ranked

### F1 (MEDIUM, ops and provenance; outside FP32_ALT's code): a LIVE X5 run is on T62's float64 solve
`rb_x35t_c0_s3601` (the X35 breakers-stage control C0, a fork of `rb_toy_ent_s4002_deep` @ 290M) is pinned at
`4609c155`, a T62-window commit (after `bd3126e0`, before `d4af9f1e`). It was live at 05:10 on 2026-10-20. Its config
is v159, and every v144+ config builds X5's hypothesis tokens, so it runs `fixed_size_tau` in float64.

This has three consequences:
- **Cost.** If the 2026-10-19/20 battery's +82–127 % of a training cycle holds on its path, this control pays it. Its
  pinned restarts keep paying it.
- **Comparisons.** An arm pinned at or after `d4af9f1e` differs from C0 in the τ solve as well as in its lever. The
  two solves compute the same real function to ~1e-6, so the scientific effect is nil. Wall-clock throughput still
  differs, and "one lever per A/B" asks for the difference to be DECLARED.
- **Provenance.** A HEAD fork or resume of this run migrates its v159 config to `tau_solve = "fp32_alt"`. Its record
  will then say fp32_alt for 290M+ steps trained on fp64. The migration comment ("no run before v160 CHOSE one … T62's
  float64 had no field") is literally true but hides this case. It is the only T62-window run in `models/`; I scanned
  every `pin_history`.

**Fix (the orchestrator decides):**
- note C0's solve on the X35 sheet;
- decide whether C0 restarts at a FP32_ALT pin (a fork may switch: §3.2);
- extend the v160 migration comment to name T62-window runs.

### F2 (MEDIUM, unverified on the production programs): Triton on CUDA is not measured
Every "two programs land on one τ" measurement compares programs on the CPU:
- ATen against the Inductor C++ kernel;
- the tests' 32 permutations.

The production pair is T2's compiled CUDA rollout (Triton) against the CUDA learner and the K9(b) probe. The proof
still predicts the result there:
- the bound holds for any order;
- a fast `exp` (`ex2.approx`, x·log2e rounding) is a ~\|x\|u perturbation of the scores, the same class as the rounding
  of x = a + τ, which the proof already absorbs.

Nobody has run it, though. The GPU battery reads cost and naive-vs-alt agreement, not compiled-vs-eager τ on CUDA.

**Fix:** parametrize `test_fp32_alt_compiled_by_inductor_matches_eager_on_the_tau_class_row` over `device` and run
the cuda case (marked `gpu`, ~2 min) in the next leased window, before X35's production GO. The live run's K9(b)
record is the field confirmation.

### F3 (LOW, documentation): the proof's constant is understated by the reduction depth
"c ≈ (depth + 2)u, depth ≤ 9" assumes a tree reduction. The CPU Inductor kernel (`cpp.simdlen = 1`) accumulates
`tmp_acc0 = tmp_acc0 + tmp16` serially over 400 columns (read in the generated code). CUDA reductions also run a
serial per-thread prefix before the tree.

The provable worst case is therefore c ≤ (n + 1)u, which bounds τ to 2c ≈ **4.8e-5**, not 1e-6. Realized errors are
random-walk sized: every sequential order I tried, including magnitude-sorted ones, gave the same 1.3e-7 – 1.3e-6
(`review/checks.log` §2). Over 300 random rows the maximum was 4.0e-6 absolute, which is 0.86 ulp of the largest
score (`review/check_random.log`).

Even the worst case is 200× below the naive band, and below K9(b)'s 1e-4 bar after the measured ∂log π/∂τ ≤ ~0.2.

**Fix:** in §3.2, the docstring, the README and the CHANGELOG, write "c ≈ (L + 2)u, L the longest serial accumulation
chain (≤ n − 1): worst case 2c ≈ 5e-5, MEASURED ≤ 4e-6".

### F4 (LOW, documentation): README F1 attributes too much to the adjacent-float wobble
It credits FP32_ALT's ~1-ulp τ wobble with "wobbles of up to about 2e-5 in log π". The first-order bound is
\|∂log π / ∂τ\| × 2 ulp(τ) ≲ 0.2 × 4e-6 ≈ 1e-6.

τ also shifts every candidate's logit by the same amount, so it can never reorder the selection. The only exception
is π values that round equal in fp32, a gap far below rule 8's 1e-6 exclusion.

I found no systematic bias. On 256 mixed rows, compiled minus eager differed on 48 rows, with a maximum of 1.9e-6, a
signed mean of +0.03 ulp(τ), and signs 29 positive to 19 negative (sign test p ≈ 0.19). The saturated subset split 4
positive to 2 negative of 145 (`review/check_compile.log`).

So the wobble cannot bias K9(b) on any row class at its 1e-4 bar or its ~2e-5 p99. The extra all-actions max
(8.4e-5 against fp64's 6.4–7.2e-5) belongs to the same fp32 program noise as the README's F4.

**Fix:** reword README F1 to "≲ 1e-6 by first order; the all-actions max difference is ordinary fp32 program noise".

### F5 (LOW): the underflow limit is real, documented inconsistently, and unreachable
The design doc and README say "~165 nats"; the module docstring says "~175". Measured: the error stays at 2.5e-6
up to 165, then reaches 0.28 at 170.

No reachable input gets there:
- the species prior spans ≲ 13 nats (marginal floor 1e-4, lift ±4);
- δ_θ would need a ~150-nat output, against a measured lead of ~23;
- OTHER's move mixture is a clamped `logit` (`HIDDEN_SLOT_MIX_EPS`).

T62's fp64 failed first: 12.6 nats off at a 100-nat lead.

**Fix:** make the docstring say "~165". No guard is needed.

### F6 (INFO; pre-existing, identical under all three solves): non-finite or absurd CANDIDATE scores
- A −inf candidate gives τ = +inf and π = NaN on it. A +inf candidate gives τ = −inf. A NaN candidate gives NaN.
- A finite candidate below ~−1e13 widens the bracket past what 64 halvings resolve: at −1e30, Σπ = 2 for k = 1.

Every producer floors or clamps its scores, so none of this is reachable. NON-candidates are safe whatever they hold
(now pinned by a test).

**Fix:** none needed. Optionally state the bracket's width precondition (≲ 1e13) next to "≤ 45.2 nats MEASURED".

### F7 (INFO): bf16 scores
The solve promotes to fp32 (now pinned by a test), but τ is returned in `scores.dtype`, so a bf16 caller gets τ
rounded to bf16 (0.025 here). No production caller passes bf16: the trunk's bf16 region ends at each round's
output, before δ_θ and every belief head. No action is needed.

### F8 (INFO): the gradient path outside the BCE
`--move-set-closure on` (`hypothesis_tokens.py:316`, OFF in production) carries π = σ(a + τ) into a non-BCE loss with
τ detached. That is not the exact implicit gradient under any solve. It is a pre-existing design choice, and
FP32_ALT moves it by ~1e-6 relative.

### F9 (INFO): compile safety
The generated Inductor C++ (CPU) keeps the expression as written:
- `abs`, then `1/(1+exp(·))`, then a select;
- two accumulators, the float residual and the int64 \|H\|;
- then `resid > float(k − |H|)`.

There is no reassociation into `resid + |H| > k`, which would reintroduce the cancellation, and no fast-math flag in
`CPU_INDUCTOR_OPTIONS`. The CUDA (Triton) code was not inspected (F2).

## Tests: what pins what, and what I added

The existing tests catch the important regressions:
- **back to the naive sum:** the accuracy test fails, and its precondition asserts that the naive sum misses;
- **fp64 as the default:** the default-value test and the traced-dtype test fail;
- **a call site that skips the model's solve:** the `_SolveTap` test fails.

They did not pin three claims, so I added them (`tau_solve_test.py`, 9 cases). Each fails when its claim is reverted
in a scratch copy, and each passed when restored:

| test | pins | reverted, it fails |
|---|---|---|
| `test_fp32_alt_is_accurate_on_small_and_full_candidate_sets_with_k_up_to_six` | "whatever the slope" on the MOVE group's small candidate sets (n = 2–8 of 400, k = n − 1 or 4: most terms in H, k − \|H\| near 0) and a k = 6 species row. The naive sum misses these by 1.2e-2 to **4.9 nats** | the naive direction test: 5 / 5 fail |
| `test_a_non_candidate_never_reaches_tau_whatever_its_score` | NaN, ±inf, ±1e30 and −1e9 in masked columns give the bit-identical τ, under both solves | `s_c = s` (no −inf mask): fails |
| `test_bf16_scores_are_solved_in_fp32_and_only_tau_is_rounded` | the docstring's "in the scores' dtype, at least fp32" | `solve_dt = scores.dtype`: 2 / 2 fail |

**Still missing, needs a GPU:** the CUDA case of the compiled-vs-eager test (F2).

## Files (`review/`, CPU only, under `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `taskset -c 5-7,13-15`)

| script | what it does |
|---|---|
| `checks.py` (helpers in `checks_lib.py`) | random rows; worst-case sequential accumulation; edge cases (k = 0, k = n, n = 0, n = 1, ties, x = 0 at the root, non-finite inputs); bf16; the underflow sweep |
| `check_random.py` | the largest-error rows, in absolute terms and in ulp of the largest score |
| `check_small.py` | small-n saturated rows (the move shape) and the bf16 return |
| `check_compile.py` | 256 mixed rows: Inductor CPU (`compile_on`) against eager, signed bias, and the set-BCE gradient under each solve |

Each script writes a `.log` beside it. The `run_battery` / GPU numbers are not reproduced here.
