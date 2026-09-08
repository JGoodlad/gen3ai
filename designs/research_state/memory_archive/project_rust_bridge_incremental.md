---
name: project_rust_bridge_incremental
description: rust sim_bridge readiness for --use-bridge=rust training — TWO wedge bugs FIXED (O(N³) replay Tier-2 93f05fb; Struggle/PP-exhaustion loop 0d73d06) AND the turn-0 CONSTRUCTION window now MODELED (Phase B, gen3_turn0_construction_v1) so the bridge is byte-for-byte with node on speed-TIED leads + unspecified-gender teams (the last parity gap); scope is bridge-only, committed goldens untouched
metadata: 
  node_type: memory
  type: project
  originSessionId: fe9aa832-f12b-4c5b-8209-a8b8c4ee882a
---

> **Archived 2026-09-08** — rust bridge wedges fixed; src/rust_sim/CLAUDE.md is the always-current doc of record. Preserved verbatim; nothing below is current.

Derisking `--use-bridge=rust` for training (2026-07-22) surfaced a HIGH-severity wedge: the multi-env
`SubprocVecEnv` run silently hung at 0 FPS (episodes finished: 0). **Root cause (gdb, not an infinite
loop):** the rust `sim_bridge` (`src/rust_sim/src/bin/sim_bridge.rs` + `bridge::run_full_battle_bridge_core`)
REPLAYED-FROM-GENESIS every CHOOSE — `replay_snapshot` = `Battle::start_with_switchins` +
`run_full_battle_logged(full script)` PER BOUNDARY → O(N²)/core-call, and sim_bridge called it per-CHOOSE →
~O(N³)/battle in the decision count. Fine for short games ("cheap at bridge scale"), catastrophic when a
**staller/staller_v2 bot × untrained ≈random policy** (the DEFAULT opponent pool) makes a long battle → each
step balloons to minutes → barrier deadlock. Node never hits this (real Showdown keeps live incremental
state, O(1)/decision).

**The real debt = no resumable/serializable Battle.** Same gap also forces the search layer's JS
`serializeBattle` hack + `Battle::reseed`=`todo!()` + the construction seed-convention workaround.

**FIX (`gen3_bridge_incremental_replay_v1`, Tier 2, SHIPPED `93f05fb` 2026-07-22):**
put the resumable capability in the ENGINE. Extracted `run_full_battle`'s outer-loop body into ONE
`FullBattleDriver` stepping primitive (phase machine AwaitMove/AwaitSwitch/Ended); `run_full_battle` is now a
thin wrapper feeding the whole script to it (behavior-preserving). `BattleStream::write_line` truly
incremental (owns a live Battle + FullBattleDriver). `bridge::BridgeSession` (persistent live Battle+driver,
threads the shared &Dex — deliberately NOT a literal BattleStream wrap, since BattleStream owns its own ~16MB
Dex) advances ONE boundary per CMD; `sim_bridge`'s Session holds one across CHOOSEs. O(1)/input, O(N)/battle.
The genesis-replay core is KEPT as the reference oracle.

**Validated (mine, independent, forced-clean fresh CARGO_TARGET_DIR):** full suite 61 suites/0 FAILED; e2e
golden md5 `3155eb796cb4bf453c6053d769ba98e5` UNCHANGED (refactor byte-preserving); NEW parity test
`bridge_incremental_matches_genesis_replay` (incremental == genesis bit-for-bit over goldens + 800-decision
stall battles incl. single+double forced replacements); timing test
`timing_bridge_incremental_linear_vs_genesis_quadratic` → incremental FLAT ~12µs/step (884-decision battle in
11ms) vs genesis 615ms for the 884th CHOOSE alone; `gen_sim_bridge_diff.js` 40 battles/0 diverged vs the Node
bridge (incl. persistent reset). Watchdogs (`TURN_LOOP_ACTION_CAP`/`BATTLE_TURN_CAP`, added same effort) fire
on a TRUE runaway loop, not this (slowness) — kept as fail-loud insurance.

**Deferred TIER 3 (shaped, not built):** native `BattleState::serialize`/`deserialize` + `reseed` — retires
the search-clone JS hack + reseed todo. FullBattleDriver is plain data so it's a mechanical add.

**Full multi-env training smoke NOT run** — the user's live training (`ai_v8_03_zarch_control_0718`, launcher)
occupies the box; the wedge cause is independently proven fixed + the bridge is byte-identical to Node, so the
training path is fixed by construction. Run the full smoke when the box is free for belt-and-braces.
Relates to [[project_bridge_training_transport]], [[project_rust_sim_port]], [[feedback_quota_pacing]].

---

**UPDATE 2026-07-22 — the deferred belt-and-braces smoke FINALLY ran (box free after stopping ai_v8_04),
and it found a SECOND, DISTINCT wedge bug the Tier-2 fix does NOT cover → `--use-bridge=rust` is STILL
UNSAFE as a training transport.**

**Bug: the rust sim INFINITE-LOOPS on Struggle (full PP exhaustion).** When a mon has 0 PP on every move
and must Struggle, the rust bridge emits `|-activate|<mon>|move: Struggle` and re-emits a `|request|` for
BOTH players WITHOUT executing Struggle (no typeless damage, no ¼ recoil, no turn advance) → the battle state
never changes → an endless no-progress bridge↔Python exchange. Node resolves Struggle and the game ends.
Measured on the deterministic repro: **node = 5,271 protocol lines (completes); rust = 1,048,107 lines (loops)**,
tail = `-activate Struggle` + Spikes/Struggle requests repeating forever.

- This is NOT the O(N³) replay slowness (Tier-2 made replay flat-fast, verified). Turns advance normally UNTIL
  a mon hits Struggle; the wedge is a **Struggle-EXECUTION defect**, not a perf regression. gdb on the wedged
  `sim_bridge`: steady-state blocked in `read()` on stdin (`BufReader<StdinRaw>::read_until`), caught once
  mid-`bridge::build_request → serialize_mon → String::clone` (emitting the re-request). `ps %cpu` ~18% is the
  LIFETIME average (fast early game), NOT the wedge instant — don't misread it as a spin.
- **Impact:** ~30% of RANDOM-play battles at concurrency=1 wedge (random play routinely reaches PP exhaustion);
  node 0/16, rust 3–5/16 (trajectory-dependent — the sim RNG is seeded but RandomPlayer actions weren't until
  I added `random.seed`). Production uses conc=1 (one bridge/env; eval default 1), so this DIRECTLY breaks
  training: one PP-exhausted battle → the env wedges → SubprocVecEnv barrier stalls the whole run. It IS the
  same "hung on staller battles" symptom class as the original O(N³) bug — two independent causes, same trigger
  (long PP-stall games).
- **Deterministic repro:** `seed=0`, `random.seed(0)`, two `RandomPlayer`s, pool teams `21022d30fb` vs
  `3495ef83ef` (Skarmory Struggles). Repro scripts: `tmp/rust_hang_isolate.py` (hang-rate over pairs, samples
  bridge cpu/stat on hang), `tmp/rust_hang_progress.py` (turn-freeze-vs-advance probe), `tmp/proto_diff.py`
  (node-vs-rust protocol-tail diff — THE artifact that named the bug).
- **FIX (not done — needs a focused rust cycle):** make Struggle EXECUTE in `src/rust_sim/src/turn/driver.rs`
  (`QAction::Move{ struggle: true }` path + the move engine) — typeless physical hit + ¼-of-user-maxhp recoil +
  turn advance, matching node. Then a NAMED deterministic regression test (fixed seed, a PP-exhaustion battle
  MUST complete — the [[feedback_edge_case_regression_tests]] rule) AND extend
  `src/rust_sim/harness/gen_sim_bridge_diff.js` to include long PP-stall battles (its 40 SHORT battles never hit
  Struggle → the coverage gap; [[feedback_validate_observable_bytes]]). Determinism ⇒ fixable
  ([[feedback_determinism_means_fixable]]); the node byte stream is the oracle.
- Until fixed, keep training on `--use-bridge=node` (or websocket). The z-space / exploiter work does NOT need
  rust.

---

**UPDATE 2026-07-22 (same day) — bug #2 FIXED. Root cause was NOT Struggle EXECUTION (the doc's hypothesis) —
the engine runs Struggle bit-for-bit (batch-validated by the pp_struggle pins + e2e). It was the bridge
REJECTING the Struggle CHOICE:** `bridge::resolve_choice` maps a poke-env `move <id>` name against the mon's
REAL 4-move set; a must-Struggle mon's request offers the synthetic `{"move":"Struggle","id":"struggle"}`, and
`"struggle"` is NEVER in the real moveset → `resolve_choice` returned `None` → the caller's out-of-range
fallback `Move(moves.len())` → `choice_is_legal` rejects it → the boundary never commits → the bridge
re-issues the SAME request forever. FIX (`gen3_bridge_struggle_resolve_v1`, one branch in `resolve_choice`,
mirroring the existing `move_locked` single-entry mapping): `if want == "struggle" { return Some(Choice::Move(0)) }`
— the driver then substitutes Struggle via the `must_struggle` exception. `"struggle"` is never a real slot id
so it's unambiguous; the numeric `move 1` already worked (only the NAME form poke-env sends was broken).
- **Verified:** the deterministic repro flips `rust: WEDGED (1,048,107 lines)` → `rust: COMPLETED finished=1`;
  full forced-clean `cargo test` green (61 suites, 0 fail); e2e golden md5 `3155eb796cb4bf453c6053d769ba98e5`
  UNCHANGED (byte-neutral — no existing golden sends `move struggle`).
- **Two revert-verified committed pins** in `src/rust_sim/src/bridge.rs` tests (BOTH fail on revert):
  `struggle_name_choice_resolves_to_move_zero_not_a_wedge` (the unit pin: `resolve_choice("struggle")` →
  `Move(0)`) + `a_bridge_battle_reaching_struggle_completes_not_wedges` (end-to-end: drive the incremental
  bridge to PP exhaustion via `move struggle`, assert it COMPLETES + Struggle's `[from] Recoil` runs — closes
  the harness coverage gap in the COMMITTED suite, stronger than the manual `gen_sim_bridge_diff.js`).
- **Lesson (reinforces [[feedback_validate_observable_bytes]]):** my earlier "byte-identical to Node → fixed by
  construction" for the Tier-2 ship was OVERCONFIDENT — `gen_sim_bridge_diff.js` only ran SHORT battles, so
  Struggle's live bridge path was never differenced against node. The belt-and-braces smoke I'd deferred is
  exactly what found it. UNCOMMITTED on `5bbf52c` (the resolve_choice fix + 2 pins) → ships at the next
  /gen3ai-ship. A `--pp-stall` extension to `gen_sim_bridge_diff.js` (node-oracle byte-parity on Struggle) is an
  optional belt-and-braces follow-up; the committed cargo integration test is the primary durable gate.

---

**UPDATE 2026-07-22 (Phase B) — the turn-0 CONSTRUCTION WINDOW is now MODELED
(`gen3_turn0_construction_v1`), resolving the project-wide "seed-convention deferral" for the
BRIDGE. `--use-bridge=rust` is now byte-for-byte with node on speed-TIED leads AND
unspecified-gender teams — the last known node-vs-rust bridge parity gap is CLOSED.**

The gap (documented in `src/rust_sim/CLAUDE.md`): `sim_bridge` fed poke-env's RAW `>start` seed to a
pure `advance_seed_for_construction(raw)` that modeled ONLY the turn-0 Quick Claw `random(1,5)`. So a
**speed-TIED lead** (the sim's turn-0 endTurn draws eachEvent tie-shuffles) or an **unspecified-gender
mon** (the sim draws `sample(['M','F'])` per mon at `addPokemon`) desynced — the diff harness SKIPPED
speed-tied leads + PINNED genders to dodge it, so those cases were never validated against node.

**Node's exact turn-0 model** (probe-reverse-engineered, `/tmp/probe_cons*.js`): `Battle.start()` sets
`midTurn=true` so `turnLoop` SKIPS the beforeTurn/residual inserts → NO turn-0 chip, NO `|upkeep|`. The
queue is effectively `[start-action, runSwitch(p1), runSwitch(p2)]`, each with a gen<5 runAction-tail
`eachEvent('Update')`; a lead speed TIE draws the `BattleQueue.insertChoice` tie-break `random(2)`
(during the start body) + one `shuffle(2)` per Update (+ a `WeatherChange` shuffle when a lead's ability
sets weather); then `endTurn` = the DisableMove/trap shuffles + Quick Claw. Distinct-speed leads draw
NONE of the tie shuffles. Gender: `gender = set.gender || species.gender || sample(['M','F'])` — a
UNIFORM 50/50 draw (NOT ratio-weighted) only when both are empty; genderless species carry
`species.gender:'N'` (→ no draw, `'N'`→`''` so no `|switch|` suffix).

**The FIX (elegant — reuse the real machinery, user's choice):**
- Data: `species.gender` (M/F/N/absent) → `gen3_species.json` (extractor) + `SpeciesData.gender`. Obs-
  neutral (the Python facade reads only baseStats/name/num/types; extractor_parity green).
- Engine: `BattleState::run_turn0_construction` (`turn/driver.rs`) — `draw_turn0_genders` (per-mon
  `sample`, p1-then-p2 team order, ratio'd-only) then builds the `[runSwitch,runSwitch]` queue via
  `insert_runswitch` (the insertChoice tie-break) + the start-tail `each_event_shuffle` + runs it through
  the shared `turn_loop` (each runSwitch fires the ability Start + `run_switch`'s WeatherChange shuffle +
  Update tail) + `disable_move_event_shuffle` (interleaves the DisableMove **AND Magnet Pull `onAny` trap**
  events) + Quick Claw. So weather-setter ties + trapper ties fall out by construction.
- Entry: `Battle::start_with_turn0_construction` + `BridgeSession::new_construct_turn0` (raw-seed);
  `sim_bridge` passes the RAW seed. The pure `advance_seed_for_construction` is DELETED.

**SCOPE (the good news — NOT the every-golden ripple the docs feared): BRIDGE-ONLY.** The committed
seed goldens seed at the sim's POST-construction `initSeed` (via `start_with_switchins`, untouched) and
so BYPASS construction — they + the two byte-fuzzers (`ab_replay`/`bridge_replay` genesis paths) stay
BYTE-IDENTICAL. So the A1 mirror-`[of]`-flip + bab_3_15 framing-flip allowlists REMAIN for the fuzzers
(they replay post-construction; retiring them by rewiring to raw seeds is an optional future step).

**Validated:** `tests/turn0_construction_test.rs` (6 cases: distinct / speed-tie / weather-setter-tie /
gender-sample / tie+gender / bench-genderless — all byte-for-byte vs the omniscient sim, seeds + sampled
genders M/F + weather). Live-binary `gen_sim_bridge_diff.js` (skip removed): Arena Trap ties 12/12 ok,
Magnet Pull/Shadow Tag ties 15/15 ok, 0 diverged, 0 skipped. Full `cargo test` 0 failed (e2e/bridge/seed
goldens byte-identical). BONUS: the gender sampling FILLS unspecified genders → Cute Charm no longer
fail-loud-panics on gendered teams via the bridge. Ships at the next /gen3ai-ship. Relates to
[[feedback_determinism_means_fixable]], [[project_bridge_training_transport]], [[project_rust_sim_port]].

**FOLLOW-UP (same day) — two `|request|`-serializer fixes the Phase-B randbats validation surfaced
(both real gen3ou request-parity residuals, user-requested + shipped):** (1)
`gen3_bridge_curse_request_target_v1` — a NON-GHOST holder's Curse request target is now `"self"`
(the runtime `nonGhostTarget`, the same `is_nonghost_curse` read the Pressure-PP path uses), was the
base-dex `"normal"`; the per-side `reconcile_curse_target` allowlist is now a dormant legacy no-op.
(2) `gen3_own_typed_hp_active_request_v1` — the OWNER's own bare-stored Hidden Power now shows the
TYPED name ("Hidden Power Dark 70") in the ACTIVE `active[].moves[]` request too (the sibling of the
B3 roster fix `gen3_own_typed_hp_request_roster_v1`, which only fixed `side.pokemon[].moves[]`) —
`serialize_active` resolves via `side_move_id`. Both owner-only + request-only (opp HP hiding + the
omniscient `|move|` bare-collapse BF1 untouched). Guarded by the retagged-CLEAN per-side corpus
fixtures `10`/`15` (allowlist → clean; a regression → non-`ok` → FAIL). Full `cargo test` 0 failed,
e2e md5 UNCHANGED.
