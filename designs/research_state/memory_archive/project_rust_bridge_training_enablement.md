---
name: project_rust_bridge_training_enablement
description: SOLVED (2026-07-31) — rust bridge TRAINS. Root cause was `switch <NICKNAME>` unresolvable (localized pool teams, e.g. "Airmure"=Skarmory). Plus a forfeit `|win|` fix. Rust is 1.41x faster than node with a 25x smaller child.
metadata:
  node_type: memory
  type: project
  originSessionId: fe9aa832-f12b-4c5b-8209-a8b8c4ee882a
  modified: 2026-08-05T02:22:56.909Z
---

> **Archived 2026-09-08** — rust IS the default training transport now (root CLAUDE.md); src/rust_sim/CLAUDE.md owns the detail. Preserved verbatim; nothing below is current.

🚨 **CRITICAL CORRECTION (2026-08-03, FIXED + SHIPPED `bc00d4d`): "rust bridge TRAINS" was true
MECHANICALLY but WORTHLESS as learning until today — every rust episode replayed ONE dice stream.**
A seedless START fell back to a CONSTANT seed (`DEFAULT_CONSTRUCT_SEED`), and the training seam
(`bridge_session.py`) + eval (`run_local_battles`) pass NO seed. MEASURED on the real seam with a
deterministic policy: **rust 4 episodes → 1 distinct outcome** (same turn, same reward, same obs
digest) vs **node 4 → 4**. Siblings, same root: a STRING seed was SILENTLY IGNORED (`"sodium,…"`
hashed identically to unseeded ⇒ replaying a node record on rust ran a DIFFERENT battle — GIGO), and
`resumeReseed.seed` was array-only while its sole producer emits the string form.
**⇒ `models/rustbridge_soak_final_0731` (5 ckpts, steps 278.7M→283.1M ≈ 4.4M steps) trained under
this — its learning signal is meaningless; do NOT fork from it.** The other 10 `rustbridge_*` runs
saved no checkpoints (bringup soaks), and the throughput benchmarks are unaffected (they measure
steps/sec, not learning). FIX: one seed parser for both fields accepting every form node accepts, a
MINTED `sodium,<hex>` when absent, and a LOUD `__ERR__` when present-but-unparseable, + a
producer-side guard (`utils/bridge/seed_spec.py`). Deliberately NOT passing a seed from Python — a
shared seed across N workers would correlate their dice; reproducibility comes from
resolve-and-record (`__RECON__` reports the resolved seed).
**ROOT-CAUSE LESSON: every gate touching this path was SEEDED, so the production SEEDLESS path was
untested** — the same coverage-hole shape as the forfeit wedge that shipped because a fuzz only ever
tested one impl. When a code path has a "default" branch nothing exercises, that branch is untested.

✅ **RUST IS NOW GOOD FOR POOL-TEAM TRAINING** (as of `bc00d4d` + rounds 30-38): pool teams 719/719
construct, 0 coverage errors in 1500 battles; `bridge_session_fuzz_test --impl rust` 30 eps; the
`--debug --use-bridge=rust` smoke completes. STILL node-only: `--search-teacher` (the port's
`input_log` is replay-EQUIVALENT, not byte-identical).
✅ **BOTH transports now bound the reject loop** (SHIPPED `aa1783d`,
`gen3_node_bridge_reject_bound_v1`): the node bridge got rust's
`REJECT_STREAK_CAP=8` mirror, so an action-mask/legality disagreement fails LOUD on `--use-bridge=node`
instead of hanging. Reset is on a COMMITTED decision only (`battle.inputLog.length`) — resetting on a
received re-request makes the cap unreachable, the subtle wrong version. Gate
`node_reject_bound_integration_test.py` (wedge pin HANGS pre-fix, + an over-eager-cap guard + a
reset-condition pin + a cap-lockstep check), revert-verified.
**RANDBATS UNBLOCKED (SHIPPED `90427d5`, ROUND 35 `gen3_forecast_v1`):** Forecast/Castform — the last
construction blocker — is MODELED (+ the hail/sandstorm weather-set moves + the suppressed-expiry
WeatherChange draw fix it flushed out; 8 seed-exact pins FC1-FC7 + the retired-guard control, all
first-try vs the real sim). Landed after the round-40 move guard (`fd3dc03`) as required.

✅ **RE-VERIFIED 2026-08-04 against the live tree + built binary** (not docs): full-universe census
**369 gen3-legal moves → 281 modeled, 88 fail-loud, 0 MISMODELED** (`SCAN_UNIVERSE=1 node
harness/scan_move_coverage.js`, exit 0); pool greedy set-cover tables EMPTY (nothing left to model
for the 719 loaded / 773 raw teams); `bridge_impl_parity_test` **12/12** incl. seed-form parity +
seedless distinctness; the round-40 guard confirmed BEHAVIOURALLY (a `fakeout` team panics at
construction — `strings` did NOT show the move names, so trust the behaviour test, not the binary
dump). Eval threads `bridge_impl` end-to-end; PFSP is fully transport-agnostic (`snapshot_pool.py`
has zero bridge refs) and the snapshot ladder threads `--impl`.

⚠️ **OPEN 2026-08-03 — is the "1.41× faster @48 envs" a TRAINING-FPS number?** It comes from
`bridge_impl_throughput_benchmark.py`, which measures N parallel env workers = **TRANSPORT ONLY**.
Every transport win in this project so far has been absorbed at real scale (the node bridge's 2.1×
transport → ~5% training), so treat 1.41× as an upper bound on the end-to-end effect until measured.
A first 2×2 {node,rust}×{compile off,on} matrix suggested only +3.5%/+8.6%, but that run is **VOID**:
the worktree predated `bc00d4d` (the seedless-START fixed-seed bug), so every rust episode replayed
one dice stream. Re-measuring on the fixed tree; see [[project_throughput_compile]] for the result.
Rust's OPERATIONAL case (25× smaller child, ~10 GB saved at 48 envs) is unaffected either way.

**GOAL (user):** make gen3ou TRAINING run on `--use-bridge=rust` with parity-or-better performance.
**RESULT: WORKS.** First PPO iteration ever completed on rust: fps 668, 0 silent stalls, real tb scalars.

## THE ROOT CAUSE — `gen3_bridge_switch_nickname_v1`
`resolve_choice`'s `SwitchSpecies` arm matched only `species_id`. **poke-env keys every mon by its
ON-FIELD IDENT (`p1a: <nickname>`) and serializes a switch as `switch <that name>`** — which for a
nicknamed mon is the NICKNAME. The gen3ou pool ships LOCALIZED teams (`data/teams/others/mcmegan/`
has `Airmure (Skarmory)`, French), so poke-env really sends `switch Airmure`. Unresolvable →
the boundary never commits → the battle stops progressing → the env's `step()` hangs until
poke-env's 120s watchdog kills the run. FIX: match NICKNAME first, then species (mirrors Showdown's
`side.chooseSwitch`, which checks `pokemon.name` first). This is the CHOICE-RESOLUTION half of
`gen3_nickname_ident_v1` — the EMISSION half (idents render the nickname) was fixed long ago and
the same insight was never carried to incoming choices. Pin:
`switch_by_nickname_resolves_like_showdown_chooseswitch`.

## THE OTHER REAL FIX — `gen3_bridge_forfeit_win_v1`
`FORCELOSE` emitted a bare `__END__` with NO `|win|`. Node writes `>forcelose` INTO the sim, so
Showdown runs `win(otherSide)` and both players get `|` + `|win|<name>`. Without it
`battle.finished` stays False forever → the next `reset()` waits on a result that can never arrive.
The training seam forfeits whenever `reset()` lands mid-battle, so EVERY episode boundary could
wedge. Fix: `BridgeSession::forfeit` mirrors the natural-end emission; `handle_forcelose` lets
`flush_new_chunks` emit the single `__END__` (calling `end_battle` again DOUBLES it — a persistent
`end_battle` resets the `ended` guard). Pin:
`a_forfeit_emits_the_win_line_to_both_sides_not_a_bare_end`.

## FAIL-LOUD GUARDS ADDED (the "stops progressing without saying so" class)
The bridge had THREE ways to stall silently. All now emit `__ERR__` naming the situation:
1. a NAME-form choice resolving to nothing (numeric out-of-range is UNTOUCHED — a legitimate,
   tested reject the forced-replacement-resume + 0-PP gates depend on);
2. `REJECT_STREAK_CAP` (8) consecutive rejects at one boundary — an RL policy is DETERMINISTIC
   given the same request, so a mask/port legality disagreement re-sends the same choice forever;
3. the R20 upstream-desync `stopped = true` path, which returned SILENTLY (fine for the OFFLINE
   replay harness that classifies via `stopped`/streams, fatal for the LIVE bridge).

## PYTHON-SIDE FIXES
- `_dispatch` RAISED on `__ERR__`, killing the reader while `_child_crashed` stayed False → silent
  hang. Now latched into `_child_error` and raised loudly at the next battle start.
  ✅ **GAP CLOSED (2026-08-04, `gen3_bridge_child_error_wakes_step_v1`)** — was: "surfaces only at
  the NEXT `reset()`; the in-flight `step()` still waits out the 120s watchdog". The reader now
  fires poke-env's EXISTING `_disconnected` signal (the one `ps_client.listen` uses for a dropped
  websocket, which `_AsyncQueue.race_get` already races against) from its two fatal exits, so a
  dead child fails the current battle immediately. **The trap:** queues bind `_disconnected` at
  CONSTRUCTION from the ORIGINAL ps_client, and `attach()` swaps in a client with its OWN fresh
  event — signalling the new client wakes NOTHING; capture the events off the QUEUES. Pinned by
  identity and revert-verified against that wrong version.
- TEARDOWN RACE: `_write_raw` wrote into a SIGKILLed child, and `close()` killed BEFORE cancelling
  the reader (cancellation is async). Now a synchronous `_closing` flag gates `_dispatch`, then
  cancel, then kill; `_write_raw` skips an exited/closing transport and swallows a racing
  `ConnectionResetError`. 2 unit tests; the fakes now model `is_closing()`/`returncode`.

## PERFORMANCE — rust WINS (the old "rust is 2x slower" note is DEAD; it was a starved box)
`src/utils/bridge/bridge_impl_throughput_benchmark.py` (N parallel env-worker PROCESSES, same
invocation, idle box):
| workers | node | rust | ratio | node RSS/child | rust RSS/child |
|---|---|---|---|---|---|
| 8 | 1852 steps/s | 2182 | **1.18x** | 225 MB | 9 MB |
| 48 (production) | 1942 steps/s | 2729 | **1.41x** | 223 MB | 9 MB |
~25x smaller child = ~0.4 GB vs ~10.7 GB of bridge children at `n_envs=48`.

## THE METHOD LESSON (cost me 3 wrong hypotheses)
I guessed `--async-rollout`, then an unresolvable MOVE name, then an upstream desync — all from
code reading + pattern-matching to the old Struggle wedge. All wrong. **The answer came in one line
the moment I recorded the actual wire traffic.** BUILD THE TRACE FIRST:
`src/utils/bridge/bridge_trace.py` — `POKESIM_BRIDGE_TRACE=<dir>` logs every OUT command / IN frame
with timestamps, per env-worker process. The wedged env's trace tail ended on
`OUT CHOOSE p2 switch Airmure` and nothing else. See [[feedback_idle_cpu_means_deadlock]].

## OPS
- Soaks MUST be DETACHED (`setsid nohup`) — the three failed 0728 soaks died with the Claude session.
- `tmp/memcap.sh <LIMIT> <name> <cmd...>` = `systemd-run --user --scope -p MemoryMax -p
  MemorySwapMax=0` (unprivileged; `ulimit -v` breaks CUDA). SwapMax=0 is load-bearing.
- A 318-byte `tb/events.out.tfevents.*` = ZERO completed PPO iterations, however many
  `🏁 Episode Finished` lines the log shows. Check the ARTIFACT.
- ✅ **MOSTLY FIXED STRUCTURALLY (2026-08-04, `gen3_contention_robust_timeouts_v1`)** — see
  [[project_contention_robust_timeouts]]. The three bridge/env wall-clock bounds now scale by
  `max(1, loadavg/cpus)` at CALL time, a timeout is its own bucket (never an "unmodeled move"
  skip), >25% timed out = INCONCLUSIVE, and every timeout message prints the load average. Full
  unit suite now PASSES at load ~30 beside a live 48-env run. Still check `uptime` before a heavy
  *benchmark* (ratios are only load-stable on an idle box) — but a suite failure is no longer
  presumptively starvation.
- **(historical) CHECK `uptime` FOR A LIVE TRAINER *BEFORE* RUNNING ANY HEAVY SUITE — not after a
  confusing failure.** Starvation manufactured failures in EVERY bridge-backed integration test,
  with a different symptom each time, so it never looked like the same bug twice:
  `bridge_impl_parity_test` counted a per-battle TIMEOUT as an "unmodeled move" skip (39/40 bogus
  skips); the prober integration tests (`falsify_scan_end_to_end`, `lookahead_is_deterministic`)
  just failed. Done THREE times, all void. Tell: the same tests PASS in isolation and PASS on the
  main checkout.
  Diagnose with `ps -eo pcpu,pid,args --sort=-pcpu | head` — a trainer/eval-worker fleet can live
  in ANOTHER agent's worktree, so `models/` under YOUR worktree looking empty proves nothing.
  Second reason to check: a heavy suite STEALS CPU from a live run you did not start.
- `pkill -f <pattern>` matches my own bash wrapper → self-kill (exit 144). Kill by explicit PID or
  stop the systemd scope.

Relates to [[project_bridge_training_transport]], [[project_rust_sim_port]], [[project_rust_bridge_incremental]].
