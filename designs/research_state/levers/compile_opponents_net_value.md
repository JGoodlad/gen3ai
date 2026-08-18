# `--compile-opponents` net value once the RECURRING bill is counted

**Status:** 🔬 open — one half measured, the other half never was · **Ledger id:** T1

One-line claim: *`--compile-opponents` was adopted on a measured +33.3% rollout FPS and a STARTUP
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
- **The benefit half** (from the existing docs, a DIFFERENT config): 406.5 → 541.8 fps at
  `--n-envs 48`, **+33.3%** marginal training FPS; B=1 CPU forward 6.53×.

## Not-known (and this is the whole lever)

- **Whether the flag is net positive at production settings.** Combining the two halves *as if the
  +33.3% transfers*: uncompiled would run 596.2 fps with no promotion bill, versus 551.4 fps
  effective compiled — **−7.5%, i.e. the flag would be a net LOSS**. Break-even is a benefit of
  about **+50%**. At +20% it is −16.7%; at +50% it is +4.1%.
- **That combination is NOT a measurement.** The +33.3% was taken on a different run config (no
  `--compile-trainer`, different arch generation), and the two numbers have never been observed in
  the same run. The sign of the net effect is genuinely unknown — this lever exists to get it.
- **n = 1 promotion** for the cost half. A second event would turn the per-run projection from a
  size into a number.

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
- **Kill (keep the flag):** compiled is faster, or the difference is within ±5%.
- Either way, record the observed promotion count per arm — an arm that got lucky on promotions is
  not a comparison.

The TD-aux rung-2 forks are NOT the vehicle (they must differ in one thing only, `--td-aux-coef`).
Run this as its own pair.
