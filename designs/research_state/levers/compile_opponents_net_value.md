# `--compile-opponents` net value once the RECURRING bill is counted

**Status:** 🔬 open — one half measured, the other half never was · **Ledger id:** T1

One-line claim: *`--compile-opponents` was adopted on a measured +43.7% rollout FPS (rust) and a STARTUP
cost model; its RECURRING cost — every self-play promotion recompiles in all N env workers — was
never measured, and at the documented benefit the two roughly cancel.*

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

  So the bill lands in ONE iteration at every realistic pool size. **Therefore the ×12.5
  extrapolation is not the optimistic reading — it is the mechanism's prediction**, and it also
  kills one candidate fix outright: **staggering must be an EXPLICIT delay, not a hoped-for
  consequence of the sampling weights**, which are nowhere near sharp enough to spread the load.

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
