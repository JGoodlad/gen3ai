# CLAUDE.md — `src/rust_sim/` (pokesim)

A from-scratch Rust reimplementation of the Pokémon Showdown battle simulator,
scoped first to **Gen 3 OU singles**, whose hard requirement is **bit-for-bit
identical** output to upstream Showdown given the same seed + teams + choices.

**`sim_bridge` is now the DEFAULT training/eval transport** (`--use-bridge` defaults to `rust`,
2026-08-14): every run is serverless unless it passes `--use-bridge node` (the A/B arm + the parity
harness) or `--use-bridge off` (websocket). See the root `CLAUDE.md` → In-process bridge transport.

The engine is **live and bit-for-bit through full battles**: every layer in the
module map below is differentially validated against the real Showdown (PRNG →
dex → team → stats → state → events → damage → turns → full battles with
switching/secondaries/status/setup/recovery/protect/spikes/phazing/leech/
substitute/explosion/fixed-damage/PP+Struggle/taunt+disable/trapping), capped by
the 220-battle real-team e2e capstone (STRICT `filtered_diverged == 0`) and a
byte-identical protocol-emission Phase 1+2+3 (132 battles / 19348 lines), and the
bridge-facing `BattleStream::write_line` streaming surface — per-write byte-gated
against the real Node `BattleStream` (44 battles / 2377 writes, `writeline_test.rs`).

### 🚨 The move census is RECOUNTED, never quoted

**Every ROUND entry below states the census as of THAT round, and each says "current" because it
was.** They are round-scoped history — do not read one as today's number. The **only** current
answer is the tool:

```bash
# THE UNIVERSE COUNT — the ENGINE is the oracle (all four gen3 universes)
cd src/rust_sim && cargo build --release --bin scan_move_probe
PROBE_KIND=move|species|item|ability ./target/release/scan_move_probe < ids.txt   # JSON verdict per id
# the POOL report + the 0-MISMODELED invariant gate — a DIFFERENT question, keep both
cd src/rust_sim/harness && SCAN_UNIVERSE=1 SCAN_UNIVERSE_LIST=1 node scan_move_coverage.js
cd src/rust_sim/harness && node scan_move_coverage.js                   # the 762-team pool report
```

🚨 **THE UNIVERSE COUNT IS THE PROBE'S TO STATE, NOT THE JS SCAN'S.** `scan_move_coverage.js`
computes its verdict from modeled-move sets **mirrored BY HAND** from `turn.rs`, and on 2026-09-08
those mirrors were three moves stale (ROUND 51's `defensecurl`, ROUND 52's `minimize` + `imprison`) —
it said 309/60 where the engine itself runs **312/57**. `scan_move_probe` cannot drift, because it
*is* the engine running. The JS scan keeps the two jobs the probe cannot do: the team-pool report and
the `0 MISMODELED` invariant gate.

**Measured 2026-09-08 by `scan_move_probe`: 369 gen3-legal moves → 312 MODELED · 57 FAIL-LOUD ·
0 MISMODELED**; **abilities 76/76 and species 392/392 are CLOSED**; **items 102/106** (the four
fail-loud: `shellbell` / `machobrace` / `mentalherb` / `mail`). The full ranked gap, by family and by
legal-learner count, is [`designs/rust_sim/gen3_coverage_census_2026-09-08.md`](../../designs/rust_sim/gen3_coverage_census_2026-09-08.md) —
a dated SNAPSHOT, not a current number. Pool **762/762** fully engine-playable (813 `.txt` files, 51 validate-fail — the
count MOVES as the pool grows, and it read 722/722 when the pool was 40 teams smaller). ⚠️ **This
is NOT the root `CLAUDE.md`'s 719-team pool and the two must not be "reconciled".** 762 is what
`Teams.import` + `TeamValidator('gen3ou')` accept out of `data/teams/*.txt`; **719** is what
`utils.team_loader.TeamLoader.get_all_teams()` returns to TRAINING (72 sample + 647 other,
measured 2026-09-07). Different filters, both current. For scale, the ROUND-40 entry says 281/88 and ROUND 44 says 286/83 —
both were true when written. **The invariant is the load-bearing claim, not the split:** 0
MISMODELED is what makes an unmodeled move a loud construction failure rather than a silent
desync, and it has held under every round separately and combined. Re-run after admitting any move
class, and fix the two always-current docs that carry a copy of this number (root `CLAUDE.md`,
`src/utils/bridge/README.md`) in the same pass — both had gone stale by 28 moves before the
2026-08-23 doc audit.

## Why "bit-for-bit" is the hard part (read this first)

Reproducing *Gen 3's mechanics* is the easy half. The constraint that costs the
effort is matching Showdown's **control flow**, for two reasons:

1. **RNG-consumption-order equivalence.** All battle randomness funnels through
   one `PRNG` (no `Math.random()` in the battle path). Bit-identity therefore
   requires consuming the RNG in the *exact same order and count* — including the
   Fisher-Yates speed-tie shuffle, which itself draws from the PRNG. Roll
   accuracy-before-crit where Showdown rolls crit-before-accuracy and every
   downstream draw desyncs. So you must mirror Showdown's `runEvent`/`singleEvent`
   dispatch order (handlers sorted by order → priority → speed), not just the
   observable formula.
2. **Byte-identical protocol output.** Our poke-env fork parses the `|...|`
   lines; they must match exactly (tokens, order, HP-fraction formatting). The
   one documented exception is `|t:|` wall-clock lines, which poke-env ignores.

The verification answer to both is **differential testing against the real
Showdown**, layer by layer. Level 1 (the PRNG) is built; see below.

## Module map

**One line per module. The unabridged Responsibility column — every type, feature signature and
draw-model note — is [`designs/rust_sim/module_map.md`](../../designs/rust_sim/module_map.md);
read the row you are about to edit.**

| Module | State | In one line |
|---|---|---|
| `prng/` | done, validated | Bit-for-bit port of `sim/prng.ts` — `Prng` over `SodiumRng` (ChaCha20, default) + `Gen5Rng` (64-bit LCG). |
| `json.rs` | done | Tiny std-only JSON reader so the dex parses with zero deps. Load-time only, never in the battle path. |
| `dex/` | done, validated | Static data over this repo's `data/pokemon/*.json` (the same source `agents.gen3_data` uses). `Dex::for_gen(3)`, move-id aliases, and the `ItemData`/`AbilityData`/`MoveData` mechanic PARAMETERS the data-driven classes fold on. |
| `team.rs` | done, validated | Bit-faithful `Teams.unpack`/`pack` for one team — the packed string the bridge feeds `>player`; ingests Showdown AND poke-env lowercase forms. |
| `stats.rs` | done, validated | Gen-3 in-battle stat computation: exact floor placement, integer nature math, the Shedinja `maxHP` hook. |
| `state.rs` | construction done, validated | `BattleState`/`SideState`/`MonState`/`Field` + every modelled volatile and side condition, and the **`cached_speed`** the tie-shuffles and residual sort read. |
| `event.rs` | switch-in done, validated | The dispatch core: `single_event_ability_start` + the reusable `speed_sort` with the **Fisher-Yates speed-tie shuffle** — the RNG-consumption crux. Wired for `>start` switch-in abilities. |
| `damage.rs` | done, validated | Gen-3 single-hit damage: `calc_damage(&DamageContext, &Dex) -> DamageResult{base, rolls:[u16;16]}`, EXPLICIT inputs, no `BattleState`. Gen3's own two-phase `modifyDamage` + the integer 4096 chain. |
| `turn.rs` (+ `src/turn/`) | validated through full battles | The turn cycle in Showdown's EXACT draw order, split across `mod {driver, moves, status_moves, secondaries, residuals, items, status, switch, speed, helpers}` (`gen3_turn_submodule_split_v1`, pure file-org, ZERO behaviour change). `run_turn` / `run_battle` / `run_full_battle`. |
| `protocol.rs` | types + emit API, validated | `Player`, `Choice`, `ProtocolLine`, and the append-only PRNG-free `ProtocolBuilder` — all fiddly `\|...\|` formatting in ONE place. |
| `battle.rs` | `write_line` DONE, validated | `BattleOptions`/`PlayerOptions`/`PackedTeam` + `Battle::start`/`start_with_switchins` + **`BattleStream::write_line`**, the streaming drop-in for `local_sim_bridge.js`. |
| `bridge.rs` | DONE, validated (in-scope corpus) | The PER-SIDE (`p1`/`p2`) streams + the `\|request\|` JSON + the HP-privacy fold, additive on top of the omniscient stream. `run_full_battle_bridge`. |
| `bin/sim_bridge.rs` | DONE, validated | The drop-in `node local_sim_bridge.js` REPLACEMENT, INCREMENTAL (`gen3_bridge_incremental_replay_v1`) — O(1) per CHOOSE, O(N) per battle. |
| `search.rs` | DONE, validated | The search + replay KERNELS (`gen3_rust_search_driver_v1`): the aux PRNG, `Record::parse`, `build_to_turn`, `resolve_turn*`, the `outcome_of`/`pre_state` renderers. Pure helpers. |
| `bin/search_driver.rs` | DONE, validated | The drop-in replacement for BOTH node offline drivers — the persistent `{id, cmd}` search server AND the one-shot `mode` replay verbs. Dispatch is on the KEY. |

## The callable surface (battle.rs) maps to the existing bridge

`battle.rs` deliberately mirrors the FIVE ways `src/utils/bridge/` already drives
Showdown, so a finished core is a drop-in:

| Bridge today (Node)                     | Rust surface |
|---|---|
| streaming battle (`local_sim_bridge.js`) | `BattleStream::write_line` |
| mid-battle RNG swap (counterfactual/search) | `Battle::reseed` |
| clone-and-branch (`State.serialize…`)    | `BridgeSession::snapshot` (a derived `Clone`; PRNG continuity is automatic) |
| search server (`search_driver.js`)       | `src/bin/search_driver.rs` over `src/search.rs` (`gen3_rust_search_driver_v1`) |
| offline replay / re-roll (`replay_driver.js`) | the SAME `src/bin/search_driver.rs`, one-shot `mode` verbs (`gen3_rust_replay_driver_v1`) — node splits the two families across two scripts, the port serves both from one binary, which is what `sim_bridge_bin.py::search_driver_spawn_argv` assumes |
| damage oracle (`damage_probe.js`)        | `Battle::new` + state accessors (TODO) |
| team pack / validate                     | out of core scope — keep as a thin shim |

When you implement the engine, keep these signatures stable; the bridge contract
is the spec.

## Clone-and-branch: the SNAPSHOT primitive (`gen3_bridge_clone_branch_v1`)

The last surface `--use-bridge=rust` was missing for the prober's SEARCH path
(`better-line`'s CRN-anchored beam, which branches a paused mid-battle state by CLONING it).
The Node search server does this with `State.serializeBattle`/`deserializeBattle`; the port's
answer is **`BridgeSession::snapshot()` — a derived `Clone`**, and the reason a byte format is
NOT needed is structural: Showdown needs one because its battle graph is full of cyclic object
references, whereas everything under `BridgeSession` (battle state, PRNG, driver phase, chunk

**PRNG continuity is automatic** (the whole dice state is `Prng`'s backend seed), and the `&Dex` is
threaded per method rather than owned, so a snapshot copies a couple of teams' worth of state rather
than ~16 MB. Why a byte format is not needed — and why `Battle::serialize`/`deserialize` are DELETED
rather than implemented — is in
[`designs/rust_sim/search_and_replay_drivers.md`](../../designs/rust_sim/search_and_replay_drivers.md).

| method | contract |
|---|---|
| `snapshot() -> BridgeSession` | the deep clone; shares NOTHING with the parent |
| `clear_chunks()` | reset the chunk stream so a branch's `chunks()` is only ITS suffix. Deliberately does NOT touch `prev_log_len` — that is a cursor into the BATTLE LOG, and resetting it would re-emit the whole battle into the next chunk |
| `request_kind(side) -> Option<RequestState>` | Showdown's `side.requestState`. `Move` / `Switch` (a forced replacement) / `Wait`; `None` when no boundary is open |
| `is_choice_done(side) -> bool` | `side.isChoiceDone()`. True also for a `Wait` side and when no boundary is open; FALSE across a REJECT (a reject re-issues the request and holds the boundary open) |
| `active_request_json(side) -> Option<&str>` | `side.activeRequest` — the EXACT bytes that went on the wire, stored (not rebuilt) at every emit site, INCLUDING the `trapped:true`+`update:true` re-request a hidden-trap reject re-issues. Cleared when the boundary closes / the battle ends |
| `battle_state() -> Option<&BattleState>` | the omniscient referee readout (HP/status/boosts/field/seed) |
| `winner() -> Option<usize>` | the winner once ended, from BOTH end paths (the driver's win check and `forfeit`, which ends via the protocol so the driver's phase never becomes `Ended`). `None` = still playing OR a gen-3 TIE — pair with `is_ended()` |

**Gate: `tests/bridge_clone_branch_test.rs`** (7 tests on a real seeded gen3 battle, every one
carrying a non-vacuity guard — without those, a snapshot of a FINISHED battle would pass while
proving nothing). Fault-injection proven.

**Not built here:** `Battle::{new,choose,ended,winner,seed,reseed}`, which stay `todo!()` because
every production path drives the engine through `FullBattleDriver` instead. The SECOND half — the
driver that consumes this API — is below.

## The offline drivers: the SEARCH server + the REPLAY family

`src/bin/search_driver.rs` over `src/search.rs` (`gen3_rust_search_driver_v1` +
`gen3_rust_replay_driver_v1`) is the byte-compatible drop-in for **both** node offline drivers, so
`utils/bridge/search_session.py` (the prober's `better_line` beam, the search teacher) and
`utils/bridge/reconstruction.py` (`better_line` / `lookahead` / `falsify`) swap `node` for the binary
with ZERO protocol change. **Dispatch is on the KEY**: a request carrying `mode` is the one-shot
REPLAY family (`replay` / `reroll` / `reroll_many`) — a BARE JSON object on stdout, **no trailing
newline**, exit **0**, or `{"error"}` + exit **1**, which is what `_run_driver` branches on before it
parses stdout; anything else runs the persistent `{id, cmd}` search loop (`open_root` /
`expand_many` / `close`).

**It already replaces node in `better_line`, and the COMPOSITION is gated** — `search_session.py`
has the `impl` switch, `search_clone_parity_fuzz_test` takes `--impl rust`, the search teacher threads
`SearchTeacherCallback(impl=...)`, and `src/main/train/search_teacher_composition_test.py` (`sim` +
`slow`, ~11 min) runs the real trainer on `--use-bridge rust` asserting >= 2 teacher cycles whose
corrections reach the AWR fold. `--use-bridge=node` remains the fallback.

| gate | needs node? | what it proves |
|---|---|---|
| `tests/search_driver_test.rs` (5 tests) | no | the aux-RNG stream vs node draw tables (EXACT f64 — a tolerance would only hide a divergence), clone independence, the `stuck` guard's exact 41 iterations |
| `tests/replay_driver_test.rs` (12 tests) | no | the one-shot dispatch + exit codes, the persistent protocol untouched, the `recorded_queues` refusal-pull, turn-1 opening on both verb families |
| `src/rust_sim/harness/search_impl_parity.py` | yes + a captured golden | the node `search_driver.js` wire output diffed field-by-field |
| `src/rust_sim/harness/replay_impl_parity.py` | yes | `node replay_driver.js` vs the binary LIVE on identical requests |

🚨 **RUN EACH PARITY GATE ON AT LEAST TWO FRESH SEEDS BEFORE CALLING IT GREEN.** A golden is three
random battles; across seven freshly generated ones the per-golden divergence count ran
1 / 0 / 0 / 6 / 8 / 0 / 6 — three of the seven would have read as a green gate, and one bug class
turned up only on the SEVENTH.

🚨 **AN ALLOWLIST ENTRY CAN OUTLIVE ITS OWN FIX AND THEN MISLEAD EVERY READER AFTER.** Two entries
here described the port as emitting no `|error|` "on a path poke-env never takes"; both halves were
false, and that path killed **two production launches at ~8 minutes**
(`gen3_locked_choice_never_rejected_v1`). Verify an allowlist claim against
`src/rust_sim/harness/search_impl_parity.py`'s `ALLOWLIST` and the code, never against prose. The
only live entry in either harness is `.error` message TEXT; the verdict, the exit code and the error
CLASS stay strict.

🚨 **`pre_state` VOLATILE NAMES ARE UNVERIFIED.** `pre_state` mirrors Node's `preState` and has no
consumer today; its `volatiles` list is a RECONSTRUCTION from the port's typed fields, and exactly ONE
name is positively verified (`substitutebroken`). Do not read a green parity run as evidence that
`choicelock` / `perishsong` / `twoturnmove` / … is spelled or timed right — `replay_impl_parity`
prints a `pre_state:nonempty-volatiles` count precisely so an all-empty record set cannot be mistaken
for coverage.

🚨 **A gen3ou repro MUST be replayed with `{format:'gen3ou', allowHiddenPower:true}`** — the probe
default is customgame, and the sim then diverges from the golden.

Full detail — the protocol, the structural simplification over Node's serialize/restart/baseline
dance, the kernel reuse, the `turn_log` split-line reconstruction, and the honest scope of
`pre_state`:
[`designs/rust_sim/search_and_replay_drivers.md`](../../designs/rust_sim/search_and_replay_drivers.md).

## The differential-gate ladder

Every layer is validated against the REAL Showdown, one rung at a time. **Each rung's port
implementation, harness construction and honest scope is
[`designs/rust_sim/differential_gates.md`](../../designs/rust_sim/differential_gates.md)** — read the
rung you are about to change. The ladder itself:

| rung | port | harness (regenerates the vectors) | `cargo test` gate | what it proves |
|---|---|---|---|---|
| **PRNG** (level 1) | `prng/` | `harness/gen_prng_vectors.js` (cross-checks a dependency-free JS re-derivation against the REAL `prng.js` value-by-value and **aborts on any mismatch**) -> `tests/vectors/prng_golden.txt`, ~2900 assertions | `tests/prng_golden.rs` | same seed ⇒ same draws — the determinism foundation |
| **Dex** | `dex/` | `harness/gen_dex_golden.py` dumps the `agents.gen3_data` facade's view | `tests/dex_test.rs` (~1500 lines) | Rust and the Python runtime agree BY CONSTRUCTION |
| **Team** | `team.rs` | `harness/gen_team_golden.js` captures `(IN, UNPACK, PACK)` triples from the REAL `Teams`, plus hand-crafted raw fixtures | `tests/team_test.rs` (24) | the packed bytes `>player` consumes. 🚨 The edge fixtures exist because an adversarial review caught FOUR real bit-parity bugs the happy-path golden missed — a regression here must stay caught |
| **Stats** | `stats.rs` | `harness/gen_stats_golden.js` reads each mon's `storedStats` + `maxhp` from an in-process omniscient `BattleStream` — the sim's OWN stats | `tests/stats_test.rs` (18 cases) | integer nature math, floor placement, the Shedinja `maxHP` hook |
| **State** | `state.rs::BattleState::start` | `harness/gen_state_golden.js` dumps all 12 mons at the first request | `tests/state_test.rs` | construction-time fields ONLY — switch-in EVENT effects belong to `event.rs` and have their own golden |
| **Turn** | `turn.rs::run_turn` | `harness/gen_turn_golden.js`, 15 scenarios x 60 seeds, capturing `SEED_BEFORE` / `SEED_AFTER` | `tests/turn_test.rs` | the **draw-ORDER+COUNT proof**: post-turn PRNG seed == the sim's across 780 (scenario, seed) rows |
| **E2E capstone** | the whole engine | `harness/gen_e2e_fuzz.js` | `tests/e2e_fuzz_test.rs` | STRICT `filtered_diverged == 0`, real teams, to game-end (below) |

🚨 **THE TURN RUNG'S THREE DRAW-COUNT SUBTLETIES.** Each is a desync if wrong, each is pinned, and
each is the kind of thing no formula-level reading of the mechanic would tell you:

- **An IMMUNE move draws ONLY accuracy** — gen-3 `tryMoveHit` resolves immunity AFTER the accuracy
  roll but BEFORE `getDamage`, so there is NO crit roll and NO damage roll. (Water/Volt Absorb
  additionally HEAL the defender `floor(maxhp/4)` at that short-circuit, draw-free, and only on a
  HIT.)
- **FAINT-SKIP** — if the first mover KOs the target, the second mover's queued move is cancelled
  (gen3 singles `cancelAction`-all) and draws NOTHING.
- **No Quick Claw on a faint** — the gen-3 end-of-turn Quick Claw `randomChance(1,5)` is drawn
  UNCONDITIONALLY of Quick Claw possession, but only if `endTurn()` completes; a faint defers it
  behind a switch request.

The harness GUARDS its own class invariant: a "distinct-speed" scenario whose actives silently TIE on
action speed (or vice versa) fails loudly at generation.

## E2E capstone: real teams, full battles, bit-for-bit (per-decision STATE+SEED+winner differential)

The closure gate: instead of constructed scenarios with hand-picked mons and scripted moves, the
capstone drives BOTH engines over **REAL Showdown-export teams** for **complete random battles to
game-end**, asserting per-decision state + status + boosts + confusion + running PRNG seed + winner
**bit-for-bit**. It is the union of every prior rung exercised on production data.

- **The generator** `harness/gen_e2e_fuzz.js` globs `data/teams/*.txt`, validates each under gen3ou,
  and from a fixed `MASTER_SEED` pairs distinct teams + a battle seed, picking a random legal choice
  per decision from a SEPARATE seeded choice-RNG — **restricted to the modeled allow/blocklist**
  (`isModeledMove` + `MODELED_ABILITIES` + `MODELED_ITEMS`). The battle FORMAT is `gen3customgame`
  (no clauses ⇒ no SetStatus handler-sort shuffle), so the Rust gate runs with `sleep_clause` OFF.
- **The gate** `tests/e2e_fuzz_test.rs::e2e_fuzz_golden_matches_showdown` seeds a `BattleState` ONCE
  at the sim's pre-first-decision PRNG state and replays the recorded choices WITHOUT re-seeding.
  🚨 **The invariant is STRICT `filtered_diverged == 0` over EVERY battle — there is no escape
  hatch.** The committed golden is **220 battles / 11825 decisions / 0 diverged**, byte-reproducible
  at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220).
- 🚨 **The per-decision assertion tallies are CLEAN-ONLY** — the loop breaks at the first divergence,
  so post-desync rows are never counted. A tally is not a coverage claim.
- 🚨 **Coverage is GATED, not reported.** The golden carries per-decision feature columns and the test
  enforces FLOORS (`status_present_rows >= 500`; `spikes` / `substitute` / `taunt` / `trapped` /
  `fixed_damage` / `batch5` decisions each `>= 50`; **no DISABLE floor — expected 0, the honest
  disclosure**). Each floor is teeth-verified by zeroing its flag. A generator STATISTIC would have
  been a coverage claim nobody could fail.
- 🚨 **The coverage taxonomy `tests/vectors/e2e_fuzz_taxonomy.txt` ranks gaps by STATIC TEAM
  COMPOSITION** — which unmodeled ability/item the paired teams CARRY — **not** by observed
  divergence cause, and it is **MOVE-LEVEL-BLIND** (it only ever picks damaging-or-switch choices).
  It does NOT gate `cargo test`: it is the measured remaining-work map, nothing more.
- **Run it:** `node src/rust_sim/harness/gen_e2e_fuzz.js` (env knobs `E2E_FILTERED_TARGET` [default
  **220**, the committed golden's size, so a plain regen reproduces it byte-for-byte],
  `E2E_UNFILTERED`, `E2E_MAX_TRIES`, `E2E_MASTER_SEED`) regenerates both vectors; then `cargo test`
  re-pins the Rust against them. The ignored helpers `e2e_diag` (categorize divergences
  SEED/STATE/FIRSTMOVER) + `e2e_trace_one` (per-decision HP/seed trace, `E2E_TRACE`/`E2E_LO`/
  `E2E_HI`) are the triage tools used to build the allow/blocklist and localize an engine bug.

The generator's construction, the four modeled move sets and the ability/item admission ladder, the
per-batch record, and every engine bug this capstone surfaced (Water/Volt Absorb heal, Intimidate vs
`onTryBoost` and vs Substitute, the residual-vs-faint ordering + the cached-`pokemon.speed` model, the
residual handler GATHER order, the Protect duration-handler trio, the forced-replacement
`updateSpeed`): [`designs/rust_sim/e2e_capstone.md`](../../designs/rust_sim/e2e_capstone.md).

## Regression tests (the pins that hold every fix)

The e2e capstone and the fuzzers are SWEEPS: they FIND real engine bugs bit-for-bit, but each repro is
BURIED in a golden that is regenerated every layer — not a STABLE, NAMED pin.
**`tests/regression_test.rs` backfills one dedicated pin per bug**, each a CONSTRUCTED scenario
(explicit hacked `gen3customgame` teams + an explicit seed + scripted choices through the public
`Battle::start_with_switchins` / `run_turn` / `run_full_battle` surface), whose NAME and doc comment
state WHICH bug it pins and what the WRONG pre-fix behaviour was.

🚨 **THE LAW: every edge case / engine bug a fuzz surfaces becomes a NAMED deterministic pin here**
(or, if the minimal repro needs an irreducibly complex board, a `# regression:`-named scenario in the
relevant golden harness), **and the pin is only real once you have REVERTED the fix and watched it
fail.** A pin that passes on the pre-fix binary is not a pin.

Two assertion styles: **STATE pins** (hp/status/boost — no PRNG fragility) and **DRAW-COUNT (seed)
pins** (the post-decision PRNG seed vs the REAL-Showdown ground truth, whose printed `seedAfter`s are
copied verbatim into the test as constants). Regenerate the ground truth after any PRNG/draw-order
change, then update the constants:

```bash
node src/rust_sim/harness/probe_regression_rng.js
node src/rust_sim/harness/probe_residual_order_rng.js
node src/rust_sim/harness/probe_phaze_regression_rng.js
```

The full bug -> pin map (64 rows), each family's ground-truth probe, and the FEATURE pins for
newly-modelled mechanics:
[`designs/rust_sim/regression_pins.md`](../../designs/rust_sim/regression_pins.md).

## A/B fuzzer (the continuous differential parity hunter)

The e2e capstone is a FIXED 220-battle committed golden; the **A/B fuzzer** is its UNBOUNDED sibling —
`harness/ab_fuzz.js` runs for hours unattended, generating fresh team pairs + seeds + random legal
choices, driving the REAL Showdown sim and the port side by side through `src/bin/ab_replay.rs`, and
saving a **self-contained, standalone-replayable repro** for every divergence. Zero API quota while
running; every future mechanic layer becomes automatically stress-tested.

- **Three team modes** (`--mode`, default `randbats`): **`randbats`** — Showdown's OWN gen3
  random-battle generator, with sets adapted at the SET level to be port-replayable (adjustment rate
  logged per chunk); **`random`** — the MODELED-UNIVERSE generator, the coverage multiplier that
  flushes out modeled-predicate ↔ engine drift (195 species / 146 distinct moves on its first smoke,
  vs the pool's 21 / 37); **`pool`** — the filter-clean `data/teams/` teams with fresh seeds/choices.
  Flags: `--battles N` / `--hours H`, `--master-seed S` (defaults from time, **ALWAYS printed** ⇒
  reproducible), `--chunk N`, `--out DIR`, `--keep-chunks`.
- **The verdict taxonomy** `ab_replay` emits per battle, in precedence order:
  `seed` [a draw bug] > `request` > `species` > `state` [hp/maxhp/fainted/left] > `status` > `boost` >
  `confusion` > `spikes` > `firstmover`. Engine PANICS are CAUGHT (`catch_unwind` + a
  message-capturing hook) and reported as `"verdict":"panic"` — the loop never dies.
- **Repros** land in `<out>/divergences/<runid>_<battleid>/` as `battle.txt` (a single-battle chunk,
  standalone FOREVER, independent of generator drift) + `summary.json`. After an engine fix the same
  `ab_replay <dir>` must flip to `ok` — **then pin it** per the law above.
- 🚨 **The tool is FAULT-INJECTION PROVEN** (2026-07-03): a dropped draw ⇒ 6/6 flagged `kind=seed`, a
  +1-damage state error ⇒ 6/6 `kind=state`, a flipped winner ⇒ 5/6 `kind=winner` (the 6th a
  never-ended prefix battle, correctly still ok). A hunter nobody has fault-injected is a hunter that
  may be finding nothing.
- 🚨 **`ab_fuzz_out*` run dirs are gitignored — NEVER commit run output.**
- **Run it:** see the README runbook ("A/B differential fuzzer"). Quick start:
  `node src/rust_sim/harness/ab_fuzz.js --mode randbats --hours 12` (overnight),
  `--mode random --battles 200 --master-seed S` (reproducible bounded hunt);
  replay any repro with `target/release/ab_replay <repro-dir>`.
- **The root-causing workhorse** is `harness/probe_repro_simtrace.js` — replay ANY saved repro dir
  through the REAL sim with per-draw PRNG call-site instrumentation.

The driver/replayer internals and every closed finding record (the first bounded smoke's fix queue,
the residual tail's 7 engine bugs, fix-queue #4's 3 more):
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

### The OMNISCIENT BYTE differential (`--protocol`, `gen3_omniscient_byte_fuzz_v1`)

`--protocol` turns the A/B fuzzer's reconstructed STATE+seed+winner check into a **literal `|...|`
protocol byte differential**: per battle it TEES the real omniscient filtered log into the chunk
golden, `ab_replay --protocol` replays via `run_full_battle_logged`, filters BOTH sides through a
shared DENYLIST (drop `debug`/`error`, normalize `|t:|`), and first-divergence-diffs to a
`kind:"protocol"` verdict — **reported ONLY after state/seed/winner match, so a draw bug still
surfaces as `seed`**. `--format {gen3customgame,gen3ou}` threads the run format (gen3ou = the
clause-shuffle draw path + the OU framing). Genders are pinned (`pinGenders`) so the sim never draws
one at construction. Isolated build + run:

```bash
CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --release --bin ab_replay
POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay \
  node src/rust_sim/harness/ab_fuzz.js --mode pool --protocol \
  --format gen3ou --battles 300
```

FAULT-INJECTION PROVEN (a mangled `[from] item: Leftovers` tag ⇒ 6/6 flagged `kind=protocol` at the
exact `-heal` line). The byte bugs it found and closed — BF1-BF4, the STATUS-MOVE EMISSION-FORM SWEEP
(pool byte-clean 27% -> ~95%, 13 pins) and the WIDE-NET round (BF-F16..BF-F20) — are recorded in
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md). All were
observation-only: the full seed suite stayed BYTE-IDENTICAL and the e2e md5 unchanged.

### The KNOWN-RESIDUAL ALLOWLIST + the GREEN GATE (`gen3_omniscient_byte_fuzz_v1`)

The byte fuzzer is a **GREEN GATE**: a NEW divergence fails loudly, while a DOCUMENTED,
non-gen3ou-impacting artifact is EXPLICITLY allowlisted — never silently ignored. **These clauses are
the live definition of when the gate may pass; edit them only with an injection proof.**

`ab_replay --protocol` classifies a first-divergence byte diff via
`classify_known_residual(golden_framing, engine_framing, leads_speed_tie)` and adds
`"allowlisted":<reason>` to the per-battle verdict ONLY when the divergence FORM matches a documented
residual; otherwise the field is ABSENT and the divergence is a hard failure. Both live entries share
one root — the unmodeled turn-0 CONSTRUCTION speed-tie Fisher-Yates shuffle, the project-wide seed
convention every committed golden depends on — and both are **seed=None-invisible, so ZERO production
impact under `--use-bridge=rust`** (at `seed=None` the port is the sole oracle, and
`event::run_start_switchins` falls back to a DETERMINISTIC side-order at a raw-Speed tie, drawing
nothing).

- **E1 `turn0-construction-speed-tie-attribution`** — a PURE PERMUTATION of identical-CONTENT framing
  lines. The classifier takes the two FULL framing WINDOWS (everything before the first `|turn|1`),
  sorts both, and allowlists ONLY IF `leads_speed_tie` AND the two windows are an **IDENTICAL
  MULTISET**. 🚨 **A CONTENT change therefore makes the multisets DIFFER ⇒ returns None ⇒ the gate
  FAILS.** That narrowing is the point: the prior coarse per-line-TYPE key SWALLOWED a
  content-different framing divergence at a mirror lead — a potential real bug.
- **A1 `turn0-construction-speed-tie-mirror-of-flip`** (`classify_construction_mirror_of_flip`) — the
  single-line weather-`[of]`-FLIP on a same-species MIRROR lead, which is NOT a pure permutation (one
  line's `[of]` CONTENT changed) so E1 returns None. Allowlisted **ONLY when ALL SIX STRUCTURAL
  CLAUSES hold**, else None ⇒ the gate FAILS: (1) the construction speed-tie (`leads_speed_tie`);
  (2) the two equal-length framing windows differ in EXACTLY ONE line; (3) that line, in BOTH golden
  and engine, is a `-weather`/`-ability` framing line; (4) the two are byte-identical after stripping
  the trailing `|[of] pNa: <name>` clause (same weather/ability, same `[from]`); (5) the two `[of]`
  targets are the two DIFFERENT active slots (one `p1a:`, one `p2a:`); (6) both `[of]` idents map —
  via the framing `|switch|` details' species field — to the SAME species (the sibling mirror). An
  `[of]` to a non-sibling or different-species mon, a different weather/ability prefix, a
  missing/extra framing line, or a non-mirror pair breaks a clause. **GATE-INTEGRITY PROVEN** by two
  mangled-golden injections plus 7 `a1_allowlist_tests` in `ab_replay.rs`, the load-bearing one being
  `clause6_wrong_of_to_a_real_different_species_mon_fails`.

- **`ab_fuzz.js --protocol`** counts `allowlisted` SEPARATELY from `diverged`, reports
  allowlisted-by-reason, and **exits non-zero ONLY on a non-allowlisted `diverged`/`panic`/
  `parse_error`**. Allowlisted repros are saved under `<out>/allowlisted/` (auditable, and a fixture
  source); real divergences under `<out>/divergences/`.
- **`tests/byte_fuzz_corpus_test.rs`** (the `cargo test` gate) enforces "no NEW kinds": each fixture
  resolves to either `ok` (the emission-form fixtures stay byte-clean) OR a `diverged` verdict whose
  `allowlisted` reason EXACTLY equals a `# ALLOWLIST <reason>` header the fixture is tagged with. So
  (i) a residual fixture that stops matching its reason FAILS, and (ii) **nobody can add a
  silently-ignored divergence — every escape is a named allowlist entry backed by a tagged repro.**
  FAULT-INJECTION PROVEN: stripping a fixture's tag makes the corpus test FAIL. To add a fixture, drop
  a clean fuzzer repro `battle.txt` in the folder (see its `README.md`) — the test auto-discovers it.

The R- and T-numbered bugs behind the current green state (R3 IV-derived Hidden Power BP, R2 the
Leftovers `-heal` slot-condition gather order, R13 Encore x Sleep Talk, R15 Sleep Clause x a
self-Rest sleeper, T1 freeze persistence vs Hidden Power Fire, the Endure and Natural-Cure emission
fixes, and the Pressure-Curse PP root of the round-2 tail):
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

## Bridge / request A/B fuzzer (the per-side + `|request|` parity hunter)

The **PER-SIDE sibling** of `ab_fuzz.js`: `harness/bridge_ab_fuzz.js` verifies, over random teams,
that the crate's PER-SIDE (`p1`/`p2`) streams + the `|request|` JSON — the poke-env legal-action
requests, including the maybeTrapped/trapped switch-legality state machine — are BYTE-IDENTICAL to the
real Node `getPlayerStreams`. It is the validation harness for `bridge.rs`. Its taxonomy: `preamble` /
`perside` / `privacy` [HP-fold] / `request` [JSON] / `error` [trapped] / `chunk_count` / `panic`.

- **Modes** (`--mode`, default `trapping`): **trapping** — a coordinated Arena-Trap / Magnet-Pull /
  Shadow-Tag matchup vs varied grounded/Flying/Levitate/Steel/Ghost foes, the one mode where the port
  is already bit-for-bit on the omniscient stream, so ALL divergences are genuine request/per-side
  issues; plus `randbats` / `random` / `pool` reusing `ab_fuzz.js`'s exported providers. **TRAPPING
  PROBES** (`--trap-prob`, default 0.5) issue a REJECTED `switch` first so the `|error|` +
  `trapped:true` re-request round is exercised.
- 🚨 **Isolated build: `CARGO_TARGET_DIR=/tmp/pokesim_target_bridge`** — NEVER the shared `target/`,
  which holds the live `ab_replay`.
- **Fault-injection PROVEN**: drop the firm-trapped flag ⇒ `kind=request`; wrong `|error|` text ⇒
  `kind=error`; wrong HP-fold % under gen3ou ⇒ `kind=privacy`. Each caught, standalone-replayable, and
  restored byte-identical.
- **Three real Phase-1 bugs it found + FIXED** — Shadow Tag's FIRM trap (`trapped:true` on the FIRST
  request with no `maybeTrapped` phase, and a rejected switch draws `|error|` with NO re-request,
  unlike Arena Trap / Magnet Pull's `tryTrap(true)` -> `'hidden'` machine; `state::trap_is_firm`
  distinguishes them), the forced-Struggle `|-activate|<mon>|move: Struggle` OWNER-ONLY `sideupdate`
  line, and the per-side request residual.

- **Honest scope (next phase):** `randbats`/`random` modes surface PRE-EXISTING **omniscient-stream**
  gaps orthogonal to the request/per-side layer — the non-L100 `details` LEVEL display
  (`switch_details`/request `details` omit `, L84`; the port targets L100 gen3ou), a **mid-battle
  Intimidate `|-unboost|…|atk|0`** at the −6 Atk FLOOR (`turn.rs:6667` hardcodes the delta `-1` →
  always `atk|1`; the sim emits the CLAMPED-applied 0 — repro saved, probe-confirmed
  `harness` Intimidate-clamp), and the same Toxic-`[from]`/status-move/Water-Absorb clusters
  `ab_fuzz.js` already tracks. These belong to the omniscient fuzzer's fix-queue (they'd desync the
  raw stream too), not the bridge layer. `trapped:true` coverage is dense; a `gen3ou`-format trapping
  run additionally exercises the OU reframe + HP-privacy fold.

### The BANKED SPEC QUEUE — probe-settled specs

Each mechanic that reached the engine through a probe left a re-runnable oracle in `harness/` whose
header carries a SETTLED block: the draw model, the exact emission forms, the edges, and the named way
a naive implementation desyncs. 🚨 **Read the probe before implementing; do not re-derive from
source.** The eight queued specs (Safeguard / Recycle / Fake Out / Conversion / Torment / Imprison /
Weather Ball / Skill Swap) are ALL SHIPPED; the queue and the trap that made each one non-obvious are
in [`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

### The `ab_replay` SUBSEQUENCE SEED ANCHOR

`gen3_ab_replay_seed_anchor_subsequence_v1`. `align_seed_subsequence` aligns the sim's per-decision
seeds as a SUBSEQUENCE of the port's checkpoints, so a decision-boundary CHECKPOINT offset stops
reading as `kind:"seed"` and costing a full triage; the per-decision STATE checks then run at the
ALIGNED pairs, and verdicts still report the GOLDEN's decision index (what a reader greps for in
`battle.txt`).

🚨 **A sloppy anchor makes the omniscient gate VACUOUS — far worse than the artifact it removes.**
`anchor_tests` is 10 cases and **the NEGATIVES are the load-bearing half**: an injected extra draw, a
missing draw, a REORDERED pair (same values, wrong order), a port stream that ends early, and an empty
port stream must each still FAIL. `POKESIM_DUMP_SEEDS=1` prints both per-decision seed lists, so the
anchor's soundness precondition — the port's checkpoint list really is a SUPERSET of the sim's — is
CHECKABLE on any repro rather than assumed.

**THREE READINGS, AND ONLY THE THIRD IS RIGHT — the durable methodological lesson.** Within one
session this repro was called: (1) "the known segmentation artifact", asserted from a byte-clean
`POKESIM_PROTOCOL_ONLY` replay — under-evidenced; (2) "the port surfaces FEWER requests, possibly
round 26's hypothesis (b), a real draw-free legality bug" — WRONG, and the alarming one; (3) the
one-draw checkpoint offset above. What settled it was comparing draw POSITIONS **within a single
decision**. The earlier reads compared draw COUNTS across differently-scoped windows — `ab_replay`
plays the WHOLE scripted battle before it compares, so its 171-draw trace covers all 33 port
decisions, not the 5 the verdict names. **A count comparison whose two sides cover different windows
is not evidence, and it reads exactly like evidence.**

What the anchor did and did not close on `ab_41_9` (a one-draw checkpoint-placement artifact the
anchor correctly refuses to reconcile — the port's RNG consumption is CORRECT), and why anchoring
against the DRAW stream instead carries its own vacuity risk:
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

### `--mode ourandom` — "gen3ou-randbats", the fuzz surface that is actually the one we care about

`gen3_ou_random_teams_v1` (`harness/ou_random_teams.js`). The fuzzers had two team sources and
NEITHER is the training/ladder surface:

| mode | on-surface? | diverse? |
|---|---|---|
| `pool` | **yes** — the 762 real gen3ou teams | **no**: a FIXED human-built set from a narrow meta, and the committed capstone samples only 220 battles of it |
| `randbats` | **no** — non-L100 levels, curated movesets, near-uniform items | yes |
| **`ourandom`** | **yes** | **yes** |

**THE MOTIVATING MEASUREMENT.** Both bugs found on 2026-08-17 (ROUND 42's Trace/forecast, ROUND
43's Substitute/wrap) have **ZERO gen3ou-pool exposure** — 0 of 773 pool files carry Castform, 0
carry a wrap-family move. Training plays pool-vs-pool, so neither could ever have fired there.
That is the round-24 lesson ("fix bugs found on the SURFACE YOU CARE ABOUT") restated as a number,
and it is why a gen3ou-native random generator is worth having.

**Every input is Smogon-derived and already committed** (`gen3_smogon_stats.json` usage,
`gen3_teammate_priors.json`, and the move / item / ability / spread priors) — deliberately NOT
`data/teams/gen3_species_priors.json`, which is POOL-derived, and the whole point is independence from
the pool. **Legality is Showdown's verdict, not a reimplementation**: every generated team goes
through the real `TeamValidator('gen3ou')`, so the banlist and every team-building clause are enforced
by the sim. 🚨 **Coverage is DISCLOSED, not assumed** — move sampling is renormalized over
ENGINE-MODELED moves (so a generated battle ALWAYS plays to completion), and the renormalized mass is
printed in the run banner by `describeCoverage()` on every run.

The Hidden-Power closed form (restricting IVs to {30,31} PINS BP at 70, so the bit-0 pattern alone
selects the type — verified 16/16 by RECOMPUTING type and BP from the emitted IVs), the two bugs the
real data shapes caught, and the first results:
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

### The picker's own blind spots (found while measuring the above)

- **STRUGGLE is now pickable.** It is ENGINE-MODELED (`pp_struggle_test.rs` is a full
  STATE+PP+SEED+winner differential) but `isModeledMove` returns false for it, so a mon with every
  slot spent AND no switch had no pickable choice and the whole battle was DROPPED to a prefix.
  That truncated precisely the PP-exhaustion endgames — the deepest, most state-laden turns, and the
  ones gen3ou STALL teams produce most. The live per-side gate already accepted Struggle
  (`id === 'struggle' || isModeledMove(id)`); the two harnesses simply disagreed and the offline one
  was weaker. **A picker predicate that gates a test silently SHRINKS that test** — the same shape
  as ROUND 42's L100 pin.
- **The drop LABEL named an innocent bystander.** `forced-unmodeled-move:<moves[0]>` reported the
  FIRST slot in the request regardless of why the pick failed, so a drop on a mon whose first slot
  happened to be Substitute read as `forced-unmodeled-move:substitute` — and Substitute is modeled,
  so the label sent a reader hunting a bug that does not exist (it did, on 2026-08-17). It now names
  the moves that actually blocked, with a distinct `all-disabled(...)` reason. **A diagnostic that
  names an innocent bystander is worse than one that names nothing.**

### THE EXTERNAL-CONSISTENCY GATE (`gen_sim_bridge_diff.js`) — promoted to a green-gated fuzzer

`gen3_simbridge_diff_allowlist_v1`. **This is the strongest correctness gate in the project**, because
it is the only one that compares what poke-env ACTUALLY consumes, at the boundary poke-env sits on.

**Why it outranks the byte/seed gates.** `ab_fuzz`/`ab_replay` diff the OMNISCIENT log — which poke-env
never sees — and replay a FIXED recorded decision list, so they must ASSUME both engines segment
decisions identically. When they don't, you get a `kind=seed` artifact that cannot be distinguished from
a real legality bug (the round-26 finding: 12 of 13 open repros have ZERO wrong draws). This harness
instead spawns BOTH real bridges, feeds identical stdin, and **discovers boundaries live** (read a
request → choose → compare). The segmentation artifact CANNOT occur by construction, and a genuine extra
request is an unambiguous request-frame mismatch. It also covers the `|request|` JSON — a genuinely
separate observable with its own bug history (PA2's Spikes-under-Pressure PP was INVISIBLE to the
omniscient fuzzer and only diverged in the request's `pp` field).

**THE LAYERING PRINCIPLE.** per-side + request = the CONTRACT (the correctness requirement); the
omniscient log = a LOCALIZER (engine bug vs fold/serializer bug); the PRNG seed = a LEADING INDICATOR
(catches divergence before it is observable). **An outer-layer mismatch is ALWAYS a bug; an inner-layer
mismatch with a clean outer layer is NOT necessarily one** (the turn-0 construction residuals are exactly
that). Inner layers buy detection SPEED and LOCALIZATION, not correctness.

**What landed:**
- **The GREEN GATE + allowlist.** Previously any known-benign residual failed the run — a 10-battle

- **`--selftest`** — 14 gate-integrity assertions: 10 allowlist ones whose NEGATIVES are load-bearing
  (a different move; an alias PLUS a residual pp difference; a LEVEL value difference; a GENDER value
  difference; a missing move; a non-request line; identical lines), plus 4 pinning the switch-probe
  content discriminator. **Run it after ANY change to `ALLOWLIST_TRANSFORMS` or the probe path.**
- **THE DRAIN / PROBE CONTRACT.** Every wait on a child is bounded and every bound is CHECKED:
  `assertDrained` on START + per decision, content-based acceptance on the trapped switch probe
  (`probeWasAccepted`, `PROBE_MAX_MS` 750 ms), and a per-battle wall-clock budget (`BATTLE_BUDGET_MS`
  300 s) over the outer loop. 🚨 **`drain_timeouts` must stay 0** — any non-zero value means a child
  went quiet somewhere, even on a path that recovers. Env overrides for investigation:
  `SBD_DRAIN_MAX_MS`, `SBD_PROBE_MAX_MS`, `SBD_BATTLE_BUDGET_MS`, `SBD_TRACE_DRAINS=1`, and
  `POKESIM_SIMBRIDGE_TARGET` (so two concurrent investigations never share one cargo target dir).
- 🚨 **`--persistent` is MANDATORY for a soak** — ~600 battles/hr with it, ~80/hr without (each battle
  otherwise respawns a Node child that reloads the whole Showdown dist), and ~96% of wall time is the
  per-write quiescence settle rather than CPU.

**HONEST SCOPE:** cross-side p1/p2 interleaving is not asserted (a Node scheduler artifact the Python
demux does not depend on); `__RECON__` is excluded (a real rust deferral — `resumeReseed` works,
`gen3_bridge_resume_reseed_v1`, so reconstruction + search paths still require node); unmodeled moves
fail loud, so "clean" is always relative to the modeled universe.

What landed in the green gate + allowlist, and the measured throughput:
[`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

## Data-driven mechanics (the class framework)

**The strategic shift (Phase 1 landed 2026-07-03, `gen3_item_mechanics_v1`):** stop hand-modeling
items/abilities one id at a time. The A/B fuzzer's motivating find: Pink Bow / Polkadot Bow + the
4 gen4-named incenses sat in the e2e's `MODELED_ITEMS` while the port's hardcoded
`resolve_atk_stat_mods` match-arm priced NONE of them — a drift class that recurs whenever an
allow-list and an engine table are maintained by hand in two places. The framework kills the
class: extract the gen3-RESOLVED item/ability tables ONCE (like the dex), classify EVERY entry
into mechanic CLASSES with machine-readable parameters, and implement ONE generic engine path per
class, validated by one class-sweep golden.

- **THE MOD-CHAIN LAW (the Light Ball cautionary tale).** gen3 resolves through gen4 → … → base,
  and later mods REPLACE and DELETE handlers: base Light Ball doubles Atk+SpA, the gen4 mod
  REWRITES it to an `onBasePower` double, the gen3 mod REWRITES it again to **SpA-ONLY ×2**.
  NEVER regex a single data file — extract from the resolved dist; the probe/golden against the
  real sim is the only oracle. (Same law as the taunt/disable durations.)

- **The extraction, the class map and the DRIFT GATE.** `harness/dump_gen3_mechanics.js` reads the
  RESOLVED `Dex.mod('gen3')`, dumps every gen3 item and ability with its resolved handler inventory +
  extracted parameters, classifies each (**UNCLASSIFIED fails the dump**), and writes
  `tests/vectors/gen3_mechanics_inventory.md` — the class map every future phase executes against.
  **`--check` is the drift gate**: it verifies the committed `data/pokemon/gen3_items.json` /
  `gen3_abilities.json` mechanics fields EXACTLY match the resolved dist. Run it whenever either
  regenerates. `--json` emits the machine-readable extraction.

**The wired classes** — Phase 1's STAT/BP-MODIFIER item family (TYPE_BOOST, SPECIES_STAT, CHOICE),
Phase 2's ability DMG_MOD family (PINCH, unconditional Atk, Guts, Marvel Scale), Phase 3's ACCURACY
pipeline (the acc/eva stage table, ACCURACY_ITEM, ACCURACY ability), the STATUS_IMMUNE / SWITCH_OUT /
TYPE-INTERACTION classes, and ability batches 1-4 — each carry their parameters, draw model and
dedicated golden in
[`designs/rust_sim/data_driven_mechanics.md`](../../designs/rust_sim/data_driven_mechanics.md),
alongside the per-class roadmap and the committed-data contract. **Read the class before adding a
member to it.**

### Handler-completeness audit (`gen3_handler_audit_v1`) — the dispatch-bus guarantee as a STATIC gate

The port implements effects AT-SITE (no generic runEvent bus). The recurring bug class that allows: an
effect carries a handler at a hook we never enumerated, or hand-placed at the wrong site — Immunity's
onUpdate cure, Cloud Nine's onEnd WeatherChange, Plus/Minus's cross-field onModifySpA, the tox
onSwitchIn reset, sun/rain's unguarded onFieldResidual, facade's onBasePower. The audit closes the
class STATICALLY.

`harness/dump_gen3_handlers.js` enumerates EVERY handler-bearing key (`on*` functions AND the numeric
priority/order/subOrder metadata AND draw-relevant declaratives) on EVERY effect in the port's
REACHABLE surface — the MODELED ∪ NOOP abilities + MODELED items (from `gen_e2e_fuzz.js`, the one
source of truth) + every condition the engine can enter + every `isModeledMove` move + `struggle` —
each (effect, hook) row carrying an FNV-1a **body fingerprint** of the resolved source, so a semantic
change in the dist is DETECTED. `tests/vectors/gen3_handler_audit.json` is the manifest: one row per
(effect, hook) with an explicit `disposition: implemented | noop_justified | unreachable_justified |
failloud_guarded`, the fingerprint, and — for `implemented` — an **anchor** `file.rs::symbol` that
must grep in `src/`. The dispositions are CURATED CODE in `harness/handler_audit_dispositions.js`.

🚨 **The gate FAILS on all four drift modes** — a resolved key with NO manifest row (a new/unnoticed
handler), a stale row, a body FINGERPRINT drift (re-probe before re-accepting), or a dead
`implemented` anchor — and is wired into `cargo test` as `tests/handler_audit_test.rs`, which
**fails loudly if node/dist are unavailable: a silently-skipped completeness gate is no gate.**
All four failure modes are perturbation-demonstrated. Regenerate after a triage:

```bash
node src/rust_sim/harness/dump_gen3_handlers.js          # regenerate
node src/rust_sim/harness/dump_gen3_handlers.js --audit   # the gate
```

🚨 **Admitting a deferred effect to a MODELED set pulls its handlers INTO the surface**, and the gate
then demands rows for them. The deferred fail-loud universe is documented in the manifest's
`_meta.excluded_deferred`.

The audit's first run surfaced TWO REAL MISSES, both latent (zero corpus exposure), both fixed
bit-for-bit and pinned — the JUMP KICK / HIGH JUMP KICK crash (`gen3_jump_kick_crash_v1`, whose crash
`getDamage` DRAWS crit + the 16-way roll) and FREEZE CLAUSE MOD (`gen3_freeze_clause_v1`, unreachable
in the gen3customgame corpora, latent for every clause format). Detail, and the rows the audit
CONFIRMED already modelled:
[`designs/rust_sim/data_driven_mechanics.md`](../../designs/rust_sim/data_driven_mechanics.md).

## Protocol emission (level-2, Phase 1 + Phase 2): the byte-identical `|...|` stream

This is the **level-2** goal — emit the byte-identical OMNISCIENT `|...|` protocol stream our
poke-env fork parses, so the port is a drop-in behind the bridge. The engine is already
bit-for-bit RNG+state faithful; this layer is a **side output** of events that ALREADY happened.

- **The emit API** (`protocol.rs`): `ProtocolBuilder` is an **append-only, PRNG-free** line buffer on
  `BattleState` (the `log` field), with ONE sim-mirroring exception — `attr_last_move_still()`, the
  port of `Battle.attrLastMove('[still]')`, for fail forms the sim itself decides RETROACTIVELY after
  draws the announce preceded. The engine pushes lines at hook points in `turn.rs`; **all fiddly
  formatting lives in ONE place** — `MonRef` / `SideRef`, `HpStatus` (the three variants `x/y` /
  `x/y <status>` / `0 fnt`, the #1 correctness point), `Cause`, `STAT_TOKENS`. The full formatter
  inventory is in
  [`designs/rust_sim/protocol_emission.md`](../../designs/rust_sim/protocol_emission.md).
  🚨 **A `MonRef`'s IDENT name is the mon's on-field NICKNAME, never the species**
  (`turn.rs::display_name` = the packed set's `set.name`, falling back to the species only when the
  set has no nickname — mirroring `Pokemon.name = set.name || species.name`). poke-env keys each mon
  by that `p<N>a: <nick>` token, so rendering the species there makes poke-env fail to match the mon
  it already tracks and try to ADD a 7th — the localized/nicknamed-team overflow CRASH
  (`gen3_nickname_ident_v1`, pinned by
  `regression_test::nicknamed_mon_renders_nickname_in_every_ident_not_species`). The SPECIES name
  (`turn.rs::species_name`) lives ONLY in the `|switch|`/`|drag|` DETAILS field.
  **Disabled by default** (`ProtocolBuilder::new()` → off): `run_full_battle` never enables it, so the
  seed suite keeps an empty, cost-free buffer AND every emit hook is a no-op that touches nothing.
  `run_full_battle_logged` enables it, emits the framing, runs the SAME `run_full_battle`, and returns
  `(BattleOutcome, Vec<ProtocolLine>)`.

- **OBSERVATION-ONLY (the load-bearing guarantee).** Emission draws NO PRNG and mutates no
  asserted state, so wiring it changes NO seed assertion. THE PROOF: the ENTIRE existing seed suite
  (`battle_test`'s 2034 cross-turn seed assertions, `fullbattle` 2053, `secondary`, the `e2e_fuzz`
  STRICT gate, every move layer, every regression pin — at the Phase-2 landing that was e2e 14228 +
  22 pins; the CURRENT tree is e2e 11673 + 44 pins, the corpus/pin growth from later layers) stays
  green with BYTE-IDENTICAL seed counts after Phase 2 — run the full suite before/after and diff (it does). The only
  engine-behaviour changes are the two emission-line REORDERS (the `|turn|N+1` marker moved to the
  next-turn top; the weather chip reads the shuffle permutation) — both provably state-/seed-
  invariant (the shuffle already drew; distinct/saturating mons) — which the seed suite re-confirms.

- **The per-phase line inventory** — which lines Phase 1 / 2 / 3 emit, in what order, and the
  deferral record (`DEFERRED_SCENARIOS` is EMPTY, 0 battles skipped) — is
  [`designs/rust_sim/protocol_emission.md`](../../designs/rust_sim/protocol_emission.md).
- **The byte-differential gate** (`tests/protocol_test.rs`): replays the capture golden through
  `run_full_battle_logged`, filters BOTH the golden's lines and the engine's output to the gated types
  (only `debug` + `error` dropped from BOTH; `|t:|` normalized), and asserts BYTE-EQUALITY per line, in
  order, with a first-divergence panic. A TRUNCATED golden (the capture hit a decision/turn cap
  mid-stall) is asserted as a byte-exact PREFIX of the longer engine output. **Result: 132 battles,
  19348 lines byte-equal, across all 22 scenarios.** The formatters are ALSO pinned by deterministic
  unit gates in `protocol.rs` — including the **disabled-builder-emits-nothing invariant**, which is
  what keeps the seed suite's buffer cost-free.

- **The drop-in endgame — BUILT** (`gen3_writeline_stream_v1`): `battle.rs`'s
  **`BattleStream::write_line`** accepts the bridge's command stream (`>start` / `>player pN` /
  `>pN move K|switch N`) and returns, PER WRITE, exactly the omniscient chunk the real Node
  `BattleStream` flushes for that write — gated by **`tests/writeline_test.rs`** against the
  per-write capture `harness/gen_writeline_capture.js` (the SAME 19-scenario corpus at 2 fresh
  seeds: **44 battles / 2377 writes / 7276 filtered lines, all chunks byte-equal**). Chunk
  attribution (probe-verified): `>start` → `|t:|`+`|gametype`; each `>player` → its `|player|`
  line (the second also the whole framing through `|turn|1`); a choice write → nothing until the
  boundary completes, then the whole turn chunk ENDING with the eager `|turn|N+1` (the sim's
  `makeRequest` flush — the port now emits the marker at turn END and the batch separator+`|t:|`
  at the COMMIT, concatenation-identical, chunk-correct). Internals + honest scope (replay-from-
  genesis; the pre-first-decision seed convention; request frames/privacy fold out of scope) are
  on the `battle.rs` module row above. Design: `PROTOCOL_EMISSION_DESIGN.md`; line grammar:
  `tests/vectors/protocol_inventory.md`.

## Standing lessons from the coverage rounds

The rounds themselves are CLOSED and live in
[`designs/rust_sim/port_build_log.md`](../../designs/rust_sim/port_build_log.md). These are the
lessons that outlive them — each cost a round to learn, and each binds the NEXT change.

- 🚨 **A new mechanic can FALSIFY an old "no draw here" proof.** When a class gains a second member,
  re-read every "this can never tie" argument: two Safeguards TIE at residual order 4, which is
  invisible to any single-side test.
- 🚨 **A determinism-oriented suite systematically UNDER-TESTS the nondeterministic default.** Every
  gate here is seeded, so the seedless bridge path had no test at all — and it ran on a FIXED seed,
  replaying one dice stream for every training episode, for months. **At least one gate must run with
  the reproducibility knob OFF and assert a DISTRIBUTIONAL property.**
- 🚨 **A gate that exempts a file by BASENAME exempts every file with that name.** Compare the
  relative path.
- 🚨 **A PICKER PREDICATE THAT GATES A TEST SILENTLY SHRINKS THAT TEST.** Struggle was fully modelled
  and fully gated, yet `isModeledMove` returned false for it — so every PP-exhaustion endgame, the
  deepest and most state-laden turns gen3ou stall teams produce, was dropped to a prefix.
- 🚨 **A DIAGNOSTIC THAT NAMES AN INNOCENT BYSTANDER IS WORSE THAN ONE THAT NAMES NOTHING.** A drop
  label reporting the first move slot regardless of why the pick failed sent a reader hunting a bug
  that does not exist.
- 🚨 **AN ALLOWLIST ENTRY CAN OUTLIVE ITS OWN FIX**, and then it misleads every reader after. So can a
  "STILL NEEDED" note. Verify against the harness and the code, never against prose.
- 🚨 **A COUNT COMPARISON WHOSE TWO SIDES COVER DIFFERENT WINDOWS IS NOT EVIDENCE, and it reads
  exactly like evidence.** Compare draw POSITIONS within a single decision.
- 🚨 **FIX BUGS FOUND ON THE SURFACE YOU CARE ABOUT.** Two bugs found on 2026-08-17 have ZERO
  gen3ou-pool exposure — 0 of 773 pool files carry either mechanic — so neither could ever have fired
  in training. That is what `--mode ourandom` exists for.
- 🚨 **A GATE NOBODY CAN START IS INDISTINGUISHABLE FROM A GATE THAT PASSES.** Both parity harnesses
  were un-runnable for weeks after a directory move left their repo-root index behind.
- 🚨 **NEVER regex a single data file for a mechanic** — extract from the RESOLVED dist. gen3 resolves
  through gen4 → … → base, and later mods REPLACE and DELETE handlers (the Light Ball cautionary
  tale). See the MOD-CHAIN LAW above.

## Conventions

- **std-only, zero dependencies.** Determinism + a no-network `cargo test` are
  the point. Add a dep only with a clear reason, and never one in the
  deterministic battle path.
- **Generation-generic.** Gen 3 OU is the only target now, but don't hard-code
  Gen-3 constants into the engine — put them behind the generation parameter
  (e.g. `Dex::for_gen(gen)`, `moves::derive_category(gen, …)`), mirroring
  Showdown's gen9→gen3 mod-delta layering, so other gens are a data layer + a
  branch, not an engine rewrite. Do not add anything that would break future gens.
- **Data source of truth** is this repo's `data/pokemon/*.json`, read the same
  way `agents.gen3_data` reads it — so Python and Rust agree by construction.

## Per-mechanic coverage records — where the detail lives

Each gen-3 mechanic below was modelled in its own coverage round, with a differential gate and
revert-verified regression pins. Those records are **CLOSED** and live verbatim in
[`designs/rust_sim/port_build_log.md`](../../designs/rust_sim/port_build_log.md) — read the one you
are debugging rather than loading all of them here. The build log also holds the 55 numbered fuzz
ROUNDS and the move-coverage BATCH 1-9 records.

Damage · Fixed-damage moves · Full battle · Multi-turn · PP tracking + Struggle · Phazing ·
Protect / Detect · Recovery moves · SNATCH · Secondary effects + onBeforeMove status · Setup moves ·
Spikes · Status moves · Switch-in events · TRICK · Taunt + Disable · Trapping · YAWN.

## Where the rest of the detail lives

| I am about to… | Read |
|---|---|
| edit a module | [`designs/rust_sim/module_map.md`](../../designs/rust_sim/module_map.md) — the unabridged per-module column |
| change a rung of the gate ladder | [`designs/rust_sim/differential_gates.md`](../../designs/rust_sim/differential_gates.md) |
| touch `search_driver` / `replay_driver` / the clone-branch API | [`designs/rust_sim/search_and_replay_drivers.md`](../../designs/rust_sim/search_and_replay_drivers.md) |
| regenerate or widen the e2e capstone | [`designs/rust_sim/e2e_capstone.md`](../../designs/rust_sim/e2e_capstone.md) |
| add or read a regression pin | [`designs/rust_sim/regression_pins.md`](../../designs/rust_sim/regression_pins.md) |
| triage a fuzz divergence, or ask whether a class was already closed | [`designs/rust_sim/ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md) |
| add an item/ability to a mechanic class | [`designs/rust_sim/data_driven_mechanics.md`](../../designs/rust_sim/data_driven_mechanics.md) |
| add a protocol line | [`designs/rust_sim/protocol_emission.md`](../../designs/rust_sim/protocol_emission.md) |
| debug ONE gen-3 mechanic | [`designs/rust_sim/port_build_log.md`](../../designs/rust_sim/port_build_log.md) — the closed round that modelled it |
