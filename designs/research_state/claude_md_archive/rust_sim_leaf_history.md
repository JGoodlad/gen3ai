# `src/rust_sim/CLAUDE.md` — prose superseded by the 2026-09-08 split

**HISTORY. Additive only — never updated.** (`designs/research_state/claude_md_archive/` is the
archive tree; the always-current homes are `src/rust_sim/CLAUDE.md` and `designs/rust_sim/`.)

On **2026-09-08** the rust_sim leaf was split from 2,659 lines / 329 KB to a command card, with the
topic detail lifted into `designs/rust_sim/` and the closed coverage rounds already living in
`designs/rust_sim/port_build_log.md` (moved 2026-09-07).

Most of the leaf moved **verbatim**. A handful of passages were instead **rewritten** in place — the
same facts, restated as a rule / a command / a hazard, with dated measurement and provenance dropped.
This file preserves those passages exactly as they read before the rewrite, so nothing the leaf ever
said is unrecoverable. **Do not read any of it as current**: where it disagrees with
`src/rust_sim/CLAUDE.md` or a `designs/rust_sim/` topic doc, those win.

---

## Passages rewritten rather than moved

### The E2E capstone's opening paragraph (leaf lines 594-600)

## E2E capstone: real teams, full battles, bit-for-bit (per-decision STATE+SEED+winner differential)

This is the closure: instead of constructed scenarios with hand-picked mons + scripted moves, the
capstone drives BOTH engines over **REAL Showdown-export teams** for **complete random battles to
game-end**, asserting per-decision state + status + boosts + confusion + running PRNG seed + winner
**bit-for-bit**. It is the union of every prior layer exercised on production data.


### The E2E capstone's `Run it` bullet (leaf lines 890-896)

- **Run it:** `node src/rust_sim/harness/gen_e2e_fuzz.js` (env knobs `E2E_FILTERED_TARGET` [default
  **220**, the committed golden's size, so a plain regen reproduces it byte-for-byte], `E2E_UNFILTERED`,
  `E2E_MAX_TRIES`, `E2E_MASTER_SEED`) regenerates both vectors; then `cargo test` re-pins the Rust
  against them. The ignored helpers `e2e_diag` (categorize divergences SEED/STATE/FIRSTMOVER) +
  `e2e_trace_one` (per-decision HP/seed trace, `E2E_TRACE`/`E2E_LO`/`E2E_HI`) are the triage tools used
  to build the allow/blocklist + localize the engine bugs.


### The regression-test section's opening + the two assertion styles (leaf lines 897-915)

## Regression tests (edge cases the e2e fuzz found — pinned deterministically)

The e2e capstone is a SEED-SWEEP over 220 random battles: it FINDS real-team-only engine bugs
bit-for-bit, but each repro is BURIED in the golden (regenerated every layer) — not a STABLE,
NAMED pin. **`tests/regression_test.rs` backfills a DEDICATED, NAMED, SELF-DOCUMENTING regression
test per such bug** — each a CONSTRUCTED scenario (explicit hacked `gen3customgame` teams + an
explicit seed + scripted choices via the public `Battle::start_with_switchins` / `run_turn` /
`run_full_battle` harness, in the style of `tests/residual_faint_test.rs`), so a future change can
never SILENTLY reintroduce the bug. The PRACTICE: **every edge case / engine bug the fuzz surfaces
becomes a dedicated deterministic test here** (or, if the minimal repro needs an irreducibly complex
board, a `# regression:`-named scenario in the relevant golden harness). Each test's NAME + doc
comment states WHICH bug it pins + the WRONG (pre-fix) behaviour; every test was verified a TRUE
PIN by REVERTING its fix and confirming the assertion fails.

Two assertion styles: **STATE pins** (hp/status/boost — no PRNG fragility) and **DRAW-COUNT (seed)
pins** (the post-decision PRNG seed vs the REAL-Showdown ground truth captured by
`harness/probe_regression_rng.js` + `harness/probe_residual_order_rng.js`, whose printed
`seedAfter`s are copied verbatim into the test as constants). The bug → pin map:


### The A/B fuzzer's opening paragraph (leaf lines 1110-1118)

## A/B fuzzer (the continuous differential parity hunter)

The e2e capstone is a FIXED 220-battle committed golden; the **A/B fuzzer** is its UNBOUNDED
sibling — a harness that runs for **hours unattended**, generating fresh team pairs + seeds +
random legal choices, driving the REAL Showdown sim and the port side-by-side, and saving a
**self-contained, standalone-replayable repro** for every divergence. Zero API quota while
running; every future mechanic layer becomes automatically-stress-tested. It found real gaps in
its first bounded smoke (see the fix queue below).


### The `--protocol` byte differential's opening paragraph + isolated build (leaf lines 1285-1306)

### The OMNISCIENT BYTE differential (`--protocol`, `gen3_omniscient_byte_fuzz_v1`)

The A/B fuzzer above checks a RECONSTRUCTED per-decision STATE tuple + seed + winner. `--protocol`
turns it into a **literal `|...|` protocol byte differential**: per battle it TEES the REAL omniscient
filtered log (`gen_e2e_fuzz.js::runBattle` now attaches `rec.lines`, `|t:|`-normalized) into the chunk
golden (`emitBattle` appends `FMT`/`L` rows in protocol mode only — the state golden is untouched), and
`ab_replay --protocol` replays via `run_full_battle_logged`, filters BOTH sides through a shared DENYLIST
(drop `debug`/`error`, normalize `|t:|` — a superset of `protocol_test.rs`'s allowlist, so newly-emitted
batch-4c/5/6/snatch line types are diffed automatically), and first-divergence-diffs to a NEW
**`kind:"protocol"`** verdict (reported ONLY after state/seed/winner match, so a draw bug still surfaces
as `seed`). `--format {gen3customgame,gen3ou}` threads the run format (gen3ou = the clause-shuffle draw
path + the OU framing, reframed via the now-`pub bridge::reframe` before the diff). The picker is widened
to admit typed **Hidden Power** in `pool` mode (`isModeledMove(id, allowHiddenPower)` — engine models
typed HP at fixed BP 70, byte-safe for gen3ou-validated 70-BP teams; `random` mode's random IVs keep it
excluded). Genders are pinned (`pinGenders`) so the sim never draws one at construction (the switch-details
construction-window gap). Isolated build + run:
`CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --release --bin ab_replay` then
`POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay node harness/ab_fuzz.js --mode pool
--protocol --format {gen3customgame|gen3ou} --battles N`. FAULT-INJECTION PROVEN (a mangled
`[from] item: Leftovers` tag → 6/6 flagged `kind=protocol` at the exact `-heal` line; restored
byte-identical via cp-aside).


### The bridge / request A/B fuzzer's opening paragraph (leaf lines 1662-1669)

## Bridge / request A/B fuzzer (the per-side + `|request|` parity hunter)

The **PER-SIDE sibling** of `ab_fuzz.js` (which A/Bs the OMNISCIENT stream): it verifies, over
RANDOM teams, that the Rust crate's PER-SIDE (`p1`/`p2`) protocol streams + the `|request|` JSON
(the poke-env legal-action requests, incl. the maybeTrapped/trapped switch-legality state machine)
are BYTE-IDENTICAL to the real Node `getPlayerStreams`. It is the validation harness for Phase 1
(`bridge.rs`).


### The bridge fuzzer's measured trapping smoke (leaf lines 1708-1714)

- **Smoke (trapping, bounded):** ~100–200 battles → 0 request/trapping/error divergences, rich coverage
  (per 100 battles ≈ 4700 requests, ≈740 trapped:true, ≈220 maybeTrapped, ≈420 `|error|` frames, ≈86
  forceSwitch), ~6000 battles/hr. Fault-injection PROVEN (drop firm-trapped → `kind=request`; wrong
  `|error|` text → `kind=error`; wrong HP-fold % [gen3ou] → `kind=privacy`; each caught + standalone-
  replayable + restored byte-identical). OBSERVATION-ONLY: the fixes are bridge-path only (turn.rs
  untouched by the Struggle line — it rides `run_full_battle_bridge`), so e2e md5
  `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED + protocol/writeline/e2e/bridge green.

### The `ab_replay` subsequence seed anchor's opening (leaf lines 1743-1748)

### The `ab_replay` SUBSEQUENCE SEED ANCHOR — shipped, and it did NOT close the repro that motivated it

`gen3_ab_replay_seed_anchor_subsequence_v1`. Round 26 left this as the scoped next step: port
`bridge_replay.rs::anchor_seed_divergence` (round 20's A2 fix) to `ab_replay`, so a decision-boundary
CHECKPOINT offset stops reading as `kind:"seed"` and costing a full triage.


### The external-consistency gate's `--selftest` + DRAIN/PROBE contract (leaf lines 1915-1928)

  an ALIAS to a CANONICAL form, so a real value difference still fails to reconcile.
- **`--selftest`** — 14 gate-integrity assertions: 10 allowlist ones whose negatives are load-bearing (a
  different move; an alias PLUS a residual pp difference; a LEVEL value difference; a GENDER value
  difference; a missing move; a non-request line; identical lines), plus 4 that pin the **switch-probe
  content discriminator** (ROUND 30). Run after ANY change to `ALLOWLIST_TRANSFORMS` or the probe path.
- **THE DRAIN / PROBE CONTRACT (ROUND 30).** Every wait on a child is bounded and every bound is
  CHECKED: `assertDrained` on START + per-decision (round 27), content-based acceptance on the trapped
  switch probe (`probeWasAccepted`, `PROBE_MAX_MS` 750 ms), and a per-battle wall-clock budget
  (`BATTLE_BUDGET_MS` 300 s) over the outer loop. The summary reports `switch_probes_accepted` /
  `drain_timeouts` / `drain_timeouts_by_tag` — **`drain_timeouts` must stay 0**; any non-zero value means
  a child went quiet somewhere, even on a path that recovers. Env overrides for investigation:
  `SBD_DRAIN_MAX_MS`, `SBD_PROBE_MAX_MS`, `SBD_BATTLE_BUDGET_MS`, `SBD_TRACE_DRAINS=1` (log every timed-out
  drain with its call-site tag), and `POKESIM_SIMBRIDGE_TARGET` (so two concurrent investigations never
  share one cargo target dir). Gate: `probe_switch_probe_accepted.js` (deterministic, hermetic,

### The handler-completeness audit — the enumerator, the manifest and the gate (leaf lines 2374-2411)

### Handler-completeness audit (`gen3_handler_audit_v1`, 2026-07-10) — the dispatch-bus guarantee as a STATIC gate

The port implements effects AT-SITE (no generic runEvent bus). The recurring bug class that
allowed: an effect carries a handler at a hook we never enumerated or hand-placed at the wrong
site — Immunity's onUpdate cure, Cloud Nine's onEnd WeatherChange, Plus/Minus's cross-field
onModifySpA, the tox onSwitchIn reset, sun/rain's unguarded onFieldResidual, facade's
onBasePower. The audit closes the class STATICALLY:

- **The enumerator** — `harness/dump_gen3_handlers.js` reads the RESOLVED `Dex.mod('gen3')`
  (the mod-chain law) and enumerates EVERY handler-bearing key (`on*` functions AND the numeric
  priority/order/subOrder metadata AND draw-relevant declaratives like `durationCallback` /
  `duration` / a move's `secondaries`/`selfdestruct`/`neverMiss`/…) on EVERY effect in the
  port's REACHABLE surface: the MODELED∪NOOP abilities + MODELED items (gen_e2e_fuzz.js — the
  one source of truth) + every condition the engine can enter (the 6 statuses, the modeled
  volatiles, the 4 weathers, spikes, the Sleep/Freeze Clause rules; `condition` sub-objects on
  surface effects auto-join) + every `isModeledMove` move + `struggle`. Each (effect, hook) row
  carries an FNV-1a **body fingerprint** of the resolved source, so a semantic change in the
  dist is DETECTED. **664 rows** (74 abilities / 59 items / 27 conditions / 168 moves) as of
  the first run. Deterministic (byte-stable regen).
- **The manifest** — `tests/vectors/gen3_handler_audit.json` (+ the human census
  `gen3_handler_audit.md`): one row per (effect, hook) with an explicit
  `disposition: implemented | noop_justified | unreachable_justified | failloud_guarded`, the
  fingerprint, and (for `implemented`) an **anchor** `file.rs::symbol` that must grep in
  `src/`. The dispositions are CURATED CODE in
  `harness/handler_audit_dispositions.js` (per-row entries + tight class rules — e.g. every
  `berryEffect` hook → the berry engine anchors; order/priority metadata inherits its sibling
  handler's disposition). First-run census: **595 implemented / 39 noop_justified /
  30 unreachable_justified**; the deferred fail-loud universe (Forecast, Liquid Ooze, the
  OHKO/Psywave/Counter family, the protocol-line gaps) is OUTSIDE the surface by construction
  and documented in the manifest's `_meta.excluded_deferred` — admitting one to a MODELED set
  pulls its handlers INTO the surface and the gate then demands rows.
- **The gate** — `node harness/dump_gen3_handlers.js --audit` FAILS on: (a) a resolved key
  with NO manifest row (a NEW/unnoticed handler), (b) a stale manifest row, (c) a body
  FINGERPRINT drift (re-probe before re-accepting), (d) a dead `implemented` anchor. Wired
  into `cargo test` as **`tests/handler_audit_test.rs`** (fails loudly if node/dist are
  unavailable — a silently-skipped completeness gate is no gate). All four failure modes
  perturbation-demonstrated. Regenerate after a triage:
  `node src/rust_sim/harness/dump_gen3_handlers.js`.

### The per-mechanic coverage-record pointer table (leaf lines 2633-2659)

## Per-mechanic coverage records — where the detail lives

Each gen-3 mechanic below was modelled in its own coverage round, with a differential gate and
revert-verified regression pins. Those records are **CLOSED** and live verbatim in
[`designs/rust_sim/port_build_log.md`](../../designs/rust_sim/port_build_log.md) — read the one
you are debugging rather than loading all of them here.

| mechanic | its round |
|---|---|
| Damage | Damage: the single-hit-physics gate (omniscient-oracle differential) |
| Fixed-damage moves | Fixed-damage moves: the `damage:` / `damageCallback` gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential) |
| Full battle | Full battle: the to-WIN/LOSS RNG-consumption gate (per-seed PER-DECISION STATE+SEED+winner differential) |
| Multi-turn | Multi-turn: the cross-turn RNG-consumption gate (per-seed STATE+SEED differential) |
| PP tracking + Struggle | PP tracking + Struggle (the first brick of `LegalActions`): per-decision STATE+HP+STATUS+PP+SEED differential |
| Phazing | Phazing: the Roar / Whirlwind forced-random-switch gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+DRAG-SPECIES+SEED differential) |
| Protect / Detect | Protect / Detect: the stall-draw + move-block gate (per-seed PER-DECISION STATE+HP+STATUS+STALL-COUNTER+SEED differential) |
| Recovery moves | Recovery moves: the self-heal / Rest draw gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential) |
| SNATCH | SNATCH: the LAST unmodeled gen-3 status move (→ 722/722) |
| Secondary effects + onBeforeMove status | Secondary effects + onBeforeMove status: the per-move-draw-bracket gate (per-seed PER-DECISION STATE+STATUS+SEED differential) |
| Setup moves | Setup moves: the self-targeting STAT-BOOST draw gate (per-seed PER-DECISION STATE+BOOST-STAGE+SEED+first-mover differential) |
| Spikes | Spikes: the entry-hazard + side-condition gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+SEED differential) |
| Status moves | Status moves: the standalone-status-MOVE draw gate (per-seed PER-DECISION STATE+STATUS+SEED differential) |
| Switch-in events | Switch-in events: the `>start` event gate (post-switch-in sim differential) |
| TRICK | TRICK: the item-swap move (`gen3_trick_v1`) |
| Taunt + Disable | Taunt + Disable: the move-SELECTION-restriction gate (per-decision STATE+TAUNT+DISABLED-SLOT+SEED differential) |
| Trapping | Trapping: the SWITCH-legality gate (per-decision STATE+per-side-TRAPPED+SEED differential) |
| YAWN | YAWN: the delayed-sleep move (`gen3_yawn_v1`) |
