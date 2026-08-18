# `--compile-opponents` net value once the RECURRING bill is counted

**Status:** ❌ **KILLED 2026-08-17, same day it was opened** — the premise was a mis-attribution.
The flag is net **+40%**, not a wash. · **Ledger id:** T1

One-line claim (REFUTED): *`--compile-opponents` was adopted on a startup cost model, and a
recurring per-promotion recompile bill cancels its benefit.*

## Verdict — the recurring bill is ~2.7%, not ~31%

`measurements/gen14_pool_refresh_compile_cost.json` (now n=2, supersedes the n=1 version):

| event | excess | compiles | which path |
|---|---|---|---|
| iteration 22 | **+1095 s** | 48 | all *timed* (each process's FIRST compile) |
| iteration 42 | **+77 s** | 27 | all *"reused this process's validated compile"* |

**Iteration 22 was a ONE-TIME transition, not a promotion.** It is where self-play first activates:
the pool is seeded from empty, so all 48 workers simultaneously load a 41 MB checkpoint *and* pay
their process's first compile. Iteration 42 is the true steady state — **+77 s ≈ 2.7% of wall-clock,
~16 min over a 25M run**, against the ~3.9 h this lever was opened on.

Net value, using the same arithmetic that opened the lever: compiled-effective 690.4 fps vs
uncompiled 493.6 fps at the +43.7% rust benefit → **+39.9%. The flag is clearly worth keeping.**

**The caching works exactly as designed** — the challenge that reopened this was right. The shared
Inductor cache was HIT at the promotion (13 files written, versus 6600+ at run startup), and
`_COMPILE_VALIDATED` correctly put every steady-state compile on the cheap path.

**Also established while settling this (worth keeping):** eval cycles are genuinely **non-blocking** —
gen-13 ran an **1865 s** `[SELFPLAY EVAL]` cycle inside a **395 s** iteration. A long eval in the log
beside a slow iteration is a coincidence of window, not a cause.

**Do not reopen without a NEW steady-state measurement.** The 18.5 min figure is retired; if you see
it quoted anywhere, it is stale.

## Original framing, retained as the record of what was wrong

## Known (established)

- **The recurring bill is real and large.** gen-14, 2026-08-17
  (`measurements/gen14_pool_refresh_compile_cost.json`): the iteration a snapshot entered the pool
  took **1234 s against a 123.7 s baseline**, with `[CompileExtractor]` firing **exactly 48 times =
  `n_envs`**. Excess **+1110 s from ONE promotion**. gen-13 kept 12 snapshots over 25M steps, i.e.
  roughly one per 2M.
- **Effective throughput, measured on gen-14 itself:** steady-state **794.7 fps** between
  promotions, **551.4 fps** with them amortised in — **−30.6%**.
- **The mechanism is per-PROCESS and cache-proof.** The Inductor cache was WARM (`[CompilePrewarm]`
  logged 40.7 s at boot; codegen is weight-independent, so it covers every future snapshot). What
  each worker still pays is dynamo tracing + guard construction, 48 of them racing on 16 cores
  behind the `SubprocVecEnv` barrier. `--compile-opponents-preload` cannot help either — it runs in
  the forkserver before the workers exist.
- **The benefit half — use the RUST number, not the generic one.** The 2×2 matrix
  {node,rust} × {compile off,on} at `--n-envs 48` on the literal arch (2026-08-03, `a5157cc`,
  4 samples/arm, 48/48 workers compiled) measured the compile effect as **+43.7% on rust**
  (417.0 → 599.4) and +35.1% on node (413.3 → 558.5). gen-14 runs `--use-bridge rust`, so **+43.7%
  is the applicable figure**; the frequently-quoted +33.3% comes from a different config and
  understates the benefit. B=1 CPU forward is 6.53×.

## Not-known (and this is the whole lever)

- **Whether the flag is net positive at production settings — and the arithmetic says it is very
  nearly a DEAD WASH.** Compiled effective throughput is 551.4 fps (promotions amortised in).
  Uncompiled, at the rust benefit of +43.7%, would be 794.7/1.437 = **553.0 fps with no promotion
  bill** — i.e. **−0.3%**, indistinguishable. At the node benefit (+35.1%) it is −6.3%; at the
  generic +33.3%, −7.5%. **So the honest headline is not "the flag loses" but "a flag adopted for a
  measured +43.7% delivers approximately ZERO at production settings."**
- **That combination is NOT a measurement.** The +33.3% was taken on a different run config (no
  `--compile-trainer`, different arch generation), and the two numbers have never been observed in
  the same run. The sign of the net effect is genuinely unknown — this lever exists to get it.
- **n = 1 promotion** for the cost half. A second event would turn the per-run projection from a
  size into a number.

- **The MECHANISM is now pinned, and it makes the projection stronger rather than weaker.** The
  obvious hope — "the first promotion is the worst case; once the pool has many entries a new
  snapshot is sampled rarely, so adoption spreads out and the herd disperses" — is **false**, and
  measurably so. At the 2M promotion the pool held **exactly one entry**, so `sample()` returned it
  with p = 1. But `recency_weight = 0.3` with PFSP off means the newest entry's weight is only
  1.3 against 1.0, *and* each worker plays ~28 episodes per iteration (2048 steps ÷ ~72 mean turns).
  P(a worker draws the new snapshot at least once in one iteration):

  | pool size | 1 | 4 | 8 | 12 | 20 |
  |---|---|---|---|---|---|
  | p(new) per episode | 1.000 | 0.283 | 0.141 | 0.094 | 0.057 |
  | **workers compiling in iteration 1 (of 48)** | **48.0** | **48.0** | **47.4** | **45.1** | **38.8** |

  That model PREDICTS the bill lands in one iteration at every pool size. **Treat it as a
  prediction, not a finding — it is currently UNTESTED, and the only adjacent data does not
  support extrapolating from the measured event.**

  **⚠️ The measured event was the WORST CASE and must not be multiplied naively.** gen-14's
  promotion went into a pool holding **exactly one entry**, so `sample()` returned the new snapshot
  with p = 1 and all 48 workers adopted it immediately. A promotion into a MATURE pool is
  **unmeasured**. The nearest evidence points cheaper: gen-13's retained segment shows compile
  bursts of 42–48 costing only **+2.2 to +8.3 min** — but those are spread across **10–11 DISTINCT
  snapshots** and follow a launcher restart, so they measure **restart warm-up, not promotion**, and
  cannot settle the question either way. (They do price the restart itself: ~8 min per 3 h restart,
  a smaller recurring cost that the startup table above already covers in principle.)

  **Third independent reason it is an upper bound (observed 2026-08-17 22:22): gen-14 passed
  4.03M steps with the pool still at ONE entry and the compile count still 48 — no second
  promotion happened.** Promotions are gated on winning an eval cycle, so they are NOT a
  fixed every-2M event; gen-13's 12 snapshots over 25M is an average, not a schedule.

  **So the ×12.5 projection is an UPPER BOUND, not an estimate**, and the per-run "~3.9 h" should be
  read that way until a promotion into a multi-entry pool is timed. gen-14's 4M promotion is exactly
  that test and costs nothing but attention.

  One implication does survive regardless: if staggering is wanted, **it must be an EXPLICIT delay**
  — `recency_weight = 0.3` (newest weight 1.3 vs 1.0, PFSP off) is nowhere near sharp enough to
  spread the load by itself.

## Pros

- If the flag is net-negative, dropping it reclaims up to ~30% of every run's wall-clock for free —
  no architecture risk, no retrain, one flag.
- Cheap to settle: a throughput A/B, no model quality involved, and the answer is a wall-clock
  number rather than a statistical gate.
- Either outcome is worth having. A confirmation makes the flag's adoption sound for the first time
  at production settings; a refutation is a large free win.

## Cons

- The rollout speedup is real and the promotion cost is bursty — a run with FEWER promotions (a
  shorter run, a slower-promoting pool) tilts toward the flag, so the answer may be
  schedule-dependent rather than a constant.
- Dropping the flag would also lose the compile path's incidental benefits (a warm shared cache for
  eval workers), which are not sized here.
- **Two facts make a deferred/staggered compile a PURE PERF change, with no behavioural question —
  this is the main reason the third option below is attractive.** (1) Compiled and eager agree to
  **max|Δ| 5.07e-07**, so which one plays is not a modelling decision. (2) The codebase ALREADY
  ships eager as the accepted degradation: `_eager_fallback_on_error` silently runs one opponent
  eager when its compile fails, and `--compile-opponents-strict` exists precisely because that
  fallback is otherwise invisible. So "serve eager for k iterations, compile later" changes no
  policy that is not already in force on the failure path.
- **The cheap path is already taken, so there is no easy win left inside the current design.**
  `_COMPILE_VALIDATED` is a process-global that skips the eager-vs-compiled TIMING forwards on every
  compile after the first in that process — the promotion path is already the cheap branch, and it
  still costs 18.5 min. What remains is the per-INSTANCE dynamo trace and guard construction, which
  is unavoidable while each snapshot is a distinct `nn.Module` with its own bound `forward`.
- A third option probably dominates both arms and is not tested by this A/B: **keep the flag and fix
  the promotion path** (stagger promotions across the barrier; serve the previous opponent for one
  iteration while the new one traces; or hold one compiled callable across snapshots). Settling
  on/off could entrench the wrong question.

## Next test

**Wall-clock A/B, ≥2 promotions per arm, same base checkpoint, everything else identical:** one arm
`--compile-opponents`, one without, ~4M steps each so both cross at least two promotion events.
Metric: **total wall-clock per 1M steps, promotions included** — not steady-state fps, which is the
number that hid this in the first place.

- **Go (drop the flag):** uncompiled is faster by >5% with the arms' ranges disjoint.
- **Kill (keep the flag):** compiled is faster by >5%, disjoint.
- **The LIKELY outcome, given the arithmetic above, is neither — a wash inside ±5%.** Pre-register
  what that means now, so it is not decided after the fact: **a wash is a verdict, and the verdict
  is that on/off is the WRONG question.** It says the ~31% promotion bill has eaten a real +43.7%
  win, and the money is in the third option — fix the promotion path (stagger promotions across the
  barrier; serve the previous opponent for one iteration while the new one traces; or hold one
  compiled callable across snapshots) and recover most of the +43.7% instead of choosing which half
  to forfeit.
- Either way, record the observed promotion count per arm — an arm that got lucky on promotions is
  not a comparison.

The TD-aux rung-2 forks are NOT the vehicle (they must differ in one thing only, `--td-aux-coef`).
Run this as its own pair.
