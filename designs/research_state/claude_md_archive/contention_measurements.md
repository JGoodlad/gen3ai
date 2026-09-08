# Contention-scaled timeouts — the measurements

Lifted verbatim from the root `CLAUDE.md` § *Running beside a live training run* on **2026-09-07**,
when that section was cut from 72 lines to ~40. The rules stayed in `CLAUDE.md`; this is the
evidence — the 0-vs-6 scaling table, the load-22 slowdown measurement, the starved parity run that
reported 39/40 timeouts as a clean pass, and the node-vs-rust throughput result it superseded.

---

### Running beside a live training run (`gen3_contention_robust_timeouts_v1`)

**The box normally carries a production run**, so any wall-clock timeout measures the spare
capacity as much as the code. This voided three separate investigations (the same tests passed in
isolation and passed on the main checkout; the only difference was a load average of 35 on 16
cores), and it produced one genuinely dangerous artifact: `bridge_impl_parity_test` counted a
per-battle TIMEOUT as an "unmodeled move" SKIP, so a starved run reported 39/40 bogus skips as a
**clean pass** that blamed the Rust port for the load average.

The bounds are now **scaled by measured CPU contention** (`src/utils/contention.py`): the
per-battle bridge timeout, the bridge-session battle-end timeout, **every child-REAP timeout**
(the local-battle-runner's and the bridge session's post-`END` waits, `SearchSession.close`, and
the search-teacher's two worker reaps), and the poke-env `race_get`
silent-stall watchdog all read `scale_timeout(...)` at CALL time, where the factor is
`max(1, loadavg / len(sched_getaffinity))`, clamped to 12x. **On an idle box the factor is exactly
1.0, so nothing changes** — this is a no-op until the box is actually busy.

- **A timeout is never a semantic outcome.** It gets its own bucket, and a run whose timeouts
  exceed 25% of attempted battles is declared INCONCLUSIVE rather than reported.
- **Every timeout message self-diagnoses** (`describe_contention()`): it prints the load average
  and the `ps -eo pcpu,pid,args --sort=-pcpu | head` command, so a starved failure ends the
  investigation instead of starting one.
- **`GEN3AI_TIMEOUT_SCALE=N`** forces the factor when you know the regime and don't want to wait
  out the load average's one-minute lag. **Run the suite under `GEN3AI_TIMEOUT_SCALE=6` after
  touching any of this** — it proves no test depends on a raw constant that the helper scales.
  That check caught two real ones: the hung-eval-cycle tests built a past timestamp from
  `_EVAL_CYCLE_TIMEOUT_SEC` directly, so they passed idle and failed loaded. (Both suites green:
  3978 passed at factor 1 and at factor 6.)
- **The eval cycle is the path MOST exposed to contention** — it deliberately runs concurrently
  with training, so it is under load 100% of the time, and the 30-min `_EVAL_CYCLE_TIMEOUT_SEC`
  hung-cycle bound now scales (`eval_cycle_timeout()`, shared by both callbacks). Firing early
  does not merely lose a cycle: `_abort_pending_cycle` kills the workers and collects **partial**
  results, which flow into `win_rate_vs_bots` (curriculum ramp), `win_rate_vs_pool` (promotion
  gate) and the ELO fit — and a truncated sample is whichever shards got scheduled, not a random
  subsample. The partial-coverage warning also no longer asserts "worker crash" as the cause,
  since an overrun-kill produces an identical shortfall.
- Prefer `ProgressDeadline` (bound the IDLE gap) over a total-duration cap wherever incremental
  progress is observable — contention stretches duration, but only a real wedge stops progress.
  A total-duration cap conflates the two by construction. Keep the total as an opt-in
  **livelock** backstop: `node_reject_bound_integration_test`'s pre-fix wedge emits `|error|`
  frames *forever*, so an idle-only bound would never expire there.
  **The per-battle bridge bound now works this way** (`gen3_battle_progress_deadline_v1`):
  `BattleStreamClient.feed` counts protocol chunks as the sign of life and
  `local_battle_runner._await_battle` bounds the gap between them, with `_PER_BATTLE_TIMEOUT`
  demoted to the livelock backstop. It was a duration cap until 2026-08-15, and the failure it
  produced is the canonical one: on a box saturated by a `cargo build --release`, the parity test
  scored **8 of 12 battles as timeouts plus a transport error and FAILED**, none of them wedged,
  all still emitting chunks — and passed warm with no code change. **Scaling does not rescue a
  cap**: the factor is `loadavg / cpus`, and the real slowdown of a starved subprocess is a
  multiple of it. *(Only the SEQUENTIAL path — under concurrency several battles share one client,
  so a lively neighbour would mask a wedged battle's silence.)*
- ⚠️ **A test that PRINTS a diagnostic must not print it in the format of a measurement.**
  `bridge_impl_parity_test`'s threshold unit test fed `timed_out=4, attempted=12` into the real
  warning function, so under `-s` it emitted `⚠️ small run: 4/12 battles TIMED OUT (33%)` into the
  same stream as the live series lines — indistinguishable from a starvation reading, and taken
  for one during this very investigation while every real series that run reported 0. Its label is
  now `SYNTHETIC unit-test sample`. Same family as the benchmark rule above: a number that cannot
  be told apart from a measurement will eventually be read as one.
- **Benchmarks get the OPPOSITE treatment — warn, never stretch.** A benchmark's output IS the
  measurement, so scaling its bounds just buys a confidently-reported wrong number. All five
  (`obs_build`, `trainer_turn`, `bridge_impl_throughput`, `bridge_heap_growth`,
  `bridge_vs_websocket_latency`) now call `warn_if_contended()` at entry and print a loud
  "THE BOX IS BUSY" banner. This is a recorded failure, not a hypothetical: a node-vs-rust
  throughput result (node 798 vs rust 427 fps) was measured on a saturated box and had to be
  superseded — **with the conclusion reversed** — and nothing in its output said so.

**Measured** (a since-deleted scratch script, 40 CPU burners → load ~47 on 16 cores, factor ~3.7, same
battles both arms): at a 2.0 s baseline, scaling OFF = **0 completed / 6 timed out**; scaling ON =
**6 completed / 0 timed out**. At a 4.0 s baseline both arms completed — the scaling matters
exactly when the bound sits within ~2x of the real battle duration, which is where a loaded box
puts you.
