# Forecast (ROUND 35) handoff — scratch, do NOT commit

## Worktree state (verified by the main session)
- Tree COMPILES. The `forecast` fail-loud in `src/rust_sim/src/state.rs` is **STILL PRESENT** —
  keep it that way until every gate is green. Empty it LAST.
- Modified: `src/rust_sim/src/{protocol.rs, state.rs, turn.rs}`. New: `src/rust_sim/src/turn/forecast.rs`.
- Probes on disk: `harness/probe_r35_forecast_{order,reporting,edges,ties,expiry_draw}.js`.
- Two prior attempts died mid-flight on an infrastructure outage (Bash unavailable), NOT on anything
  they did wrong. Main has since moved: suite baseline on `origin/main` is now **616 passed / 0 failed**.

## ⚠️ UNVERIFIED LEAD — treat as a hypothesis, not a result
The last attempt's final message was: *"Critical finding in T1 — a real pre-existing draw
divergence. Let me pin it precisely and find the clearVolatile sites."* It died before recording
what the divergence WAS. **The main session did NOT reproduce or confirm it.** Do not write it into
a round as a finding until you have re-derived it yourself.

What `probe_r35_forecast_expiry_draw.js` currently prints (main session ran it; this is the SIM
side only, so it is not by itself a divergence):
- Every turn consumes exactly **1** draw across all scenarios (sun/hail/raindance x
  SUPPRESSED(cloudnine) / EFFECTIVE(levitate)).
- Turns 2-4 gather `[each:BeforeTurn, each:Update, each:Update, each:Update, each:Weather, each:Update]`.
- Turn 1 additionally carries `each:WeatherChange` (the set).
- **Turn 5, the EXPIRY turn, carries `each:WeatherChange` and NO `each:Weather`** — the shape most
  likely to be mismodeled, and the plausible source of the claimed divergence.
- Turn 6 (no weather) drops to `[each:BeforeTurn, each:Update x3]`.

The obvious next step is to diff those per-turn handler sequences against the port's own stream
(`POKESIM_PRNG_TRACE=1`) on the identical board, and see whether the port's expiry turn agrees.

## Reminders that already cost this project rounds
- A NEW handler can change a handler-sort TIE and add a Fisher-Yates draw. ROUND 32 found
  `trapped` + `partiallytrapped` tying at one subOrder for an extra `random(0,2)` per endTurn, in
  code that asserted such a tie "can never" happen. Check; do not assume.
- Every constructed repro needs a NON-VACUITY guard — a probe where the weather never changes, or
  where Castform never switches in, silently tests nothing (ROUND 29 lost three iterations to this).
- The forme-REPORTING surface is the reason this mechanic was deferred for months: settle with BYTES
  whether `details` in `|switch|`/`|drag|`/`|replace|` and the per-side `|request|` JSON report the
  BASE species or the FORME. That is what poke-env parses and the policy observes.
- If the reporting answer changes what the POLICY observes, implement the SIM-FAITHFUL behaviour and
  escalate the observation-side consequence as an owner decision — do not quietly pick.
