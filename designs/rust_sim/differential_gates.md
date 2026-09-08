# The differential-gate ladder — per-rung detail

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 lines -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

Every layer of the port is validated against the REAL Showdown, one rung at a time: PRNG -> dex ->
team -> stats -> state -> turn, then the full-battle capstone above them. `src/rust_sim/CLAUDE.md`
carries the ladder as a table (rung / harness / gate / what it proves) plus the Turn rung's
draw-COUNT subtleties, which are parity invariants rather than detail. This document holds each
rung's port implementation, its harness construction and its honest scope.

---

## PRNG: the bit-for-bit gate (level-1 differential test)

- **Port** lives in `prng/`. `SodiumRng::next` = ChaCha20-encrypt a 36-byte zero
  buffer (key = 32-byte seed, nonce = `"LibsodiumDRG"`, counter 0); next seed =
  bytes `[0,32)`, output = big-endian u32 of bytes `[32,36)`. `Gen5Rng` = the
  64-bit LCG (`a=0x5D588B656C078965`, `c=0x00269EC3`) over four 16-bit words.
  ChaCha20 is hand-rolled (no crate) so the whole thing is std-only and auditable.
- **Reference + harness** in `harness/`. `prng_reference.js` is a dependency-free
  JS re-derivation (the executable spec the Rust mirrors). `gen_prng_vectors.js`
  cross-checks that reference against the **real** `prng.js` value-by-value and
  **aborts on any mismatch**, then emits `tests/vectors/prng_golden.txt` —
  ~2900 self-contained assertions (each keyed on a pre-state seed string, so
  `Prng::new(seed)` reconstructs and checks one call; no JSON parser needed).
- **Rust gate**: `tests/prng_golden.rs` replays every assertion. `cargo test`
  must stay green. Regenerate vectors after any PRNG change (see README).

This proves the determinism foundation in actual Rust: same seed ⇒ same draws ⇒
the rest of the engine can be trusted to replay.

## Dex: the source-of-truth gate (data differential)

- **Loader** in `dex/`. `Dex::for_gen(3)` reads `data/pokemon/*.json` via the
  std-only [`json`] reader and exposes `species`/`moves`/`item`/`ability`/
  `nature`/`type_chart`/`learnset`, plus `Type`/`MoveCategory`/`BaseStats` and a
  `to_id` normalizer (poke-env's `to_id_str`). It reads the SAME files the Python
  runtime reads, so the only real *logic* is `moves::derive_category` (the gen ≤ 3
  type-based physical/special split) — kept behind the `gen` parameter.
- **Parity harness** `harness/gen_dex_golden.py` dumps the `agents.gen3_data`
  facade's view (every species, every move incl. the derived category + resolved
  type, the full type chart, natures, learnsets) to
  `tests/vectors/dex_golden.txt`. `tests/dex_test.rs` asserts the Rust dex
  reproduces all ~1500 lines, so Rust and the Python runtime agree by
  construction (regenerate after any data/derivation change).

## Team: the packed-string gate (Showdown differential)

- **Codec** in `team.rs`. `unpack`/`pack` mirror Showdown's `Teams.unpack`/`pack`
  (`teams.ts`) for one team. `unpack` is a **sequential field walk** (NOT a
  `split(']')` — a `]` is legal inside a nickname) and is case-insensitive, so it
  ingests both Showdown's case-preserving form AND poke-env's lowercase-id form
  (our real bridge producer); `pack` re-emits Showdown-canonical bytes for the
  `>player` consumer. Bounded, documented deviations on *malformed* input only
  (numeric width, multi-char gender) — never reachable from a validator-clean team.
- **Differential harness** `harness/gen_team_golden.js` captures, from the REAL
  Showdown `Teams`, `(IN, UNPACK, PACK)` triples for ~10 constructed sets, each
  ALSO in poke-env lowercase form, PLUS hand-crafted raw fixtures pinning the
  fiddly decodes (`]`-in-nickname, short IV field `30,`, trailing-comma moves,
  empty species). `tests/team_test.rs` asserts `unpack(IN)` == UNPACK and
  `pack(unpack(IN))` == PACK for all 24. **These edge fixtures exist because an
  adversarial review caught four real bit-parity bugs the happy-path golden
  missed — a regression here must stay caught.**

## Stats: the in-battle-stat gate (sim-truth differential)

- **Calc** in `stats.rs`. `compute_stats` mirrors Showdown's `statModify`
  (`battle.js`) exactly: `EV/4` floored, `*level/100` floored, nature applied
  AFTER `+5` as **integer** math (`floor(stat*110/100)`, never f64), HP never
  natured, and the **Shedinja `maxHP` hook** (`setSpecies` overrides HP with
  `species.maxHP`). `overflowstatmod` clamps are gen3ou-absent and omitted.
- **`maxHP` is real Showdown data**, carried end-to-end: the extractor
  (`tools/pokemon_data_extractor/sync.py` `build_species`) now passes `maxHP`
  through from the pokedex → `gen3_species.json` (only Shedinja has it) →
  `dex::SpeciesData::max_hp`. So it's data-driven and **resync-safe**
  (`extractor_parity_test` passes); the Python obs facade ignores the key.
- **Differential harness** `harness/gen_stats_golden.js` drives an in-process
  omniscient `BattleStream` (the `damage_probe.js` pattern, no server) and reads
  each mon's `storedStats` + `maxhp` — the sim's OWN stats — over 18 cases (all-0
  and max EV/IV, every nature direction, levels 100/78/5/1, Shedinja, min-IV,
  extreme bases, non-÷4 EVs). `tests/stats_test.rs` reconstructs each input via
  `team::unpack` and asserts `compute_stats` matches.

## State: the construction-time gate (sim-state differential)

- **`BattleState::start`** (`state.rs`) composes the lower layers: `team::unpack`
  both teams → `stats::compute_stats` per mon → set `hp=maxhp=stats[0]`, `status=None`,
  `boosts=[0;7]`, lead = slot 0 (gen-3 singles, no team preview), `turn=0`,
  `field.weather=None`. It runs **NO switch-in events** — `event.rs` does (below).
- **The load-bearing split:** `start` builds/asserts only *construction-time*
  fields (stats, maxhp, hp, species, level, lead). Switch-in **event** effects —
  boosts (Intimidate), `field.weather` (Sand Stream), ability `Start` — are NOT in
  `start`, even though the golden's teams (Tyranitar/Salamence/Gyarados leads) make
  them fire in the sim; they live in `start_with_switchins` (`event.rs`) and have
  their own golden (next section).
- **Differential harness** `harness/gen_state_golden.js` starts a real gen3 battle
  (omniscient `BattleStream`), reaches the first request, and dumps each of the 12
  mons' `speciesid/level/maxhp/live-hp/storedStats/lead`. `tests/state_test.rs`
  feeds the identical packed teams + seed to `Battle::start` and asserts a match.

## Turn: the single-turn RNG-consumption gate (per-seed STATE differential)

This is the layer where the **RNG-consumption-order crux** finally lands on a
production path — both moves resolved, draws consumed in Showdown's exact order +
count, validated end-to-end.

- **`BattleState::run_turn(p1_slot, p2_slot, &dex) → TurnResult`** (`turn.rs`)
  executes ONE turn where both sides use a DAMAGING move:
  1. **Order** the two move actions (priority → effective speed) by **wiring
     `event::speed_sort`** onto the action queue — the FIRST production path for its
     Fisher-Yates speed-tie shuffle DRAW (one `random(0,2)` on a priority+speed tie,
     zero on distinct speed). Effective speed is the gen-3 `getActionSpeed` OVERRIDE:
     the RAW boosted + ModifySpe stat (boost-table floor; paralysis ×0.5 =
     `floor(spe*50/100)`, the gen-3 value — NOT ×0.25), **NO `trunc(spe,13)`** (the
     base-sim path gen3 replaces), capped at 10000.
  2. Per move in resolved order: **accuracy** `randomChance(acc,100)` (SKIPPED iff
     `never_miss`) → **crit** `randomChance(1, critMult[critRatio])` (UNCONDITIONAL
     for a damaging, non-immune move — every gen-3 damaging move has `critRatio ≥ 1`;
     normal = `1/16`, the high-crit set = `1/8`) → **damage** `random(16)` selecting
     `calc_damage().rolls[r]` → **apply HP** (saturate at 0) → **faint at 0**.
  3. The gen-3 end-of-turn **Quick Claw `randomChance(1,5)`** — drawn ALWAYS
     (UNCONDITIONAL of Quick Claw possession), but ONLY if `endTurn()` completes,
     i.e. **no mon fainted this turn** (a faint defers it behind a switch request).
- **Three draw-COUNT subtleties** (each a desync if wrong, each pinned):
  - **IMMUNE move** (type-chart 0× or ability/Levitate immunity) — gen-3
    `tryMoveHit` resolves immunity AFTER the accuracy roll but BEFORE `getDamage`, so
    an immune move draws **only accuracy** (NO crit, NO damage). `run_move`
    short-circuits via `move_is_immune` before the crit roll. **Water/Volt Absorb**
    additionally HEAL the defender `floor(maxhp/4)` on the absorbed Water/Electric move
    (draw-free `onTryHit`, capped, no-op at full HP) via `apply_absorb_heal` at the
    short-circuit — an e2e-capstone fix (the bare-immune path missed the heal); Flash
    Fire's Fire-boost flag stays a deferred lesser gap.
  - **FAINT-SKIP** — if the first mover KOs the target, the second mover's queued
    move is cancelled (gen3 singles `cancelAction`-all) → it draws NOTHING.
  - **No Quick Claw on a faint** — a faint pauses for a switch before `endTurn`, so
    the trailing `randomChance(1,5)` is not drawn that turn.
- **Differential harness** `harness/gen_turn_golden.js` drives the omniscient
  `BattleStream` (no server) over **15 scenarios × 60 seeds**, submitting ONE
  damaging move per side. It captures the sim's PRNG state **right before** the turn
  (`SEED_BEFORE`) and **right after** (`SEED_AFTER`), plus per-mon post-turn
  hp/fainted and per-attacker crit/miss/moved + the first mover. `tests/turn_test.rs`
  SEEDS its `BattleState` prng with `SEED_BEFORE` (sidestepping the `>start` setup
  draws — gender `sample`, turn-1 Quick Claw — this bounded step omits), runs
  `run_turn`, and asserts, for the 13 DISTINCT-speed scenarios: (a) post-turn
  hp/fainted/crit/miss/moved match AND (b) the **post-turn PRNG seed equals
  `SEED_AFTER`** — an EXACT match across **780 (scenario,seed) rows** is the
  draw-ORDER+COUNT proof (a single extra/missing/mis-ordered draw shifts the LCG and
  the seed diverges on some seed). The 2 SPEED-TIE scenarios in `turn_test.rs` assert
  (a) + **who moved first** (the action-order shuffle's decision, exercised in BOTH
  directions across the sweep); FULL tie-cycle seed parity is now closed in
  `battle_test.rs` (below), which models the per-action `eachEvent` shuffles. The
  harness GUARDS the class invariant: a "distinct-speed" scenario whose actives
  silently TIE on action speed (or vice versa) fails loudly at generation.
- **Honest scope / deferred at the SINGLE-turn layer** (the `turn_test.rs` golden
  uses no-residual moves so its post-turn HP is a clean function of only the modeled
  move draws): the multi-turn layer below adds the per-action `eachEvent` shuffles +
  residuals; still deferred everywhere — secondary effects (the per-move
  `random(100)`), status MOVES, switching, recoil/drain HP, status `onBeforeMove`
  draws (para/sleep/freeze), Leech Seed / Wish / non-Leftovers items, Thick Club /
  non-folded stat events, and protocol-string emission. The golden uses Earthquake /
  Surf / Tackle / Hydro Pump / Megahorn / Crabhammer / Swift (never_miss) and bulky
  defenders.

