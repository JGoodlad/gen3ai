//! # pokesim
//!
//! A from-scratch, **bit-for-bit-faithful** Rust reimplementation of the Pokémon
//! Showdown battle simulator, scoped first to **Gen 3 OU singles**.
//!
//! "Bit-for-bit" means: given the same seed, teams, and choice sequence, this
//! crate must emit byte-identical protocol output to upstream Showdown. That is a
//! control-flow constraint, not just a math one — the RNG has to be consumed in
//! the exact same order and count, and the `|...|` protocol lines have to match
//! exactly (our poke-env fork parses them). See `src/rust_sim/CLAUDE.md` for the
//! full rationale and the verification strategy.
//!
//! ## Status
//!
//! - [`prng`] — **DONE & validated**. Bit-for-bit vs Showdown's `prng.js`
//!   (SodiumRNG/ChaCha20 + Gen5 LCG), pinned by `tests/prng_golden.rs`.
//! - [`dex`] — **DONE**. Loads `data/pokemon/*.json` (this repo's source of
//!   truth) and mirrors the `agents.gen3_data` facade; pinned by a parity test.
//! - [`json`] — a tiny std-only JSON reader the dex parses with (zero deps).
//! - [`team`] — **DONE & validated**. Bit-faithful port of Showdown's
//!   `Teams.pack`/`Teams.unpack` for one team (the packed string the bridge
//!   feeds `>player`); pinned by `tests/team_test.rs` against a Node golden.
//! - [`stats`] — **DONE & validated**. Gen-3 in-battle stat computation
//!   (`PokemonSet` + `Dex` → `[hp,atk,def,spa,spd,spe]`), the exact
//!   `Dex.statModify` formula (integer nature math, no `overflowstatmod`);
//!   pinned by `tests/stats_test.rs` against the sim's own computed stats.
//! - [`state`] — **DONE & validated**. The construction-time in-battle state
//!   ([`state::BattleState`]: per-mon stats/HP/status/boosts, sides, field, PRNG)
//!   built from `>start` + two packed teams via [`battle::Battle::start`]. NO
//!   switch-in events run (no Intimidate/weather/ability-Start — the event
//!   engine's job); pinned by `tests/state_test.rs` against a real-sim golden.
//!   [`state::BattleState::start_with_switchins`] (and
//!   [`battle::Battle::start_with_switchins`]) compose construction + the
//!   [`event`] switch-in sequence to also produce the POST-switch-in board.
//! - [`event`] — **switch-in DONE & validated**. The generic event-dispatch core
//!   ([`event::single_event_ability_start`] + the reusable [`event::speed_sort`]
//!   `order→priority→speed→subOrder→effectOrder` selection sort with the
//!   Fisher-Yates speed-tie shuffle — the RNG-consumption crux), wired for the
//!   `>start` switch-in abilities (Intimidate's foe-Atk drop, Sand Stream /
//!   Drizzle / Drought weather). [`state::BattleState::run_start_switchins`]
//!   fires both leads' switch-in events in raw-Speed order. Pinned by
//!   `tests/switchin_test.rs` (5 differential scenarios vs the real sim's
//!   post-switch-in boosts + weather) + `event.rs`'s `speed_sort` unit tests.
//!   Move execution / the turn loop / residuals are NOT built (later steps).
//! - [`damage`] — **DONE & validated**. The Gen-3 single-hit damage calc
//!   ([`damage::calc_damage`]): a self-contained, bit-faithful port of Showdown's
//!   shared `getDamage` base formula + the gen-3-SPECIFIC two-phase `modifyDamage`
//!   (burn-first, randomize-second-to-last — the classic ordering trap). Takes
//!   EXPLICIT inputs (the sim's own stats + types/boosts/status + the move/field +
//!   a resolved `crit`), so it needs no `BattleState`; returns the deterministic
//!   max-roll `base` + the full 16-roll spread. Pinned by `tests/damage_test.rs`
//!   against the omniscient damage oracle, with the MAX roll forced so the check is
//!   EXACT (realized HP-delta == pre-roll baseDamage), not a range.
//! - [`turn`] — **multi-turn move execution + residuals DONE & validated**.
//!   [`state::BattleState::run_turn`] resolves ONE FULL turn cycle (both sides
//!   damaging): the action-order shuffle, the per-action
//!   `eachEvent('BeforeTurn'/'Update'/'Weather')` speed-tie shuffles, each move
//!   (accuracy → crit → damage [[`damage::calc_damage`]] → HP → the deferred-faint
//!   protocol), the END-OF-TURN RESIDUALS (gen-3 residualOrder: weather chip
//!   `maxhp/16` to non-Rock/Ground/Steel, Leftovers `+maxhp/16`, the major-status
//!   DoT — burn `maxhp/8`, poison `maxhp/8`, Toxic `n/16` ramp), and the Quick Claw
//!   `randomChance(1,5)` — all consuming the PRNG in Showdown's EXACT order+count.
//!   [`state::BattleState::run_battle`] LOOPS the cycle over a scripted move
//!   sequence, stopping at the first faint. **[`state::BattleState::run_full_battle`]
//!   extends that to a FULL battle to WIN/LOSS** — voluntary SWITCHES (sort before
//!   moves, the two-switch action-order tie-shuffle), the draw-FREE gen-3 switch-in
//!   (the entrant's ability `Start`), POST-FAINT replacement (single + DOUBLE, the
//!   double's `insertChoice` splice draw), and win/loss (a side out of mons loses;
//!   both out → a gen-3 tie). It takes a [`turn::ScriptDecision`] script (per-side
//!   [`turn::Choice::Move`]/[`turn::Choice::Switch`]) and returns a
//!   [`turn::BattleOutcome`] (winner + per-decision records). Pinned by
//!   `tests/turn_test.rs` (the single-turn 780-row gate), `tests/battle_test.rs`
//!   (a per-seed CROSS-TURN STATE+SEED differential, no switching), AND
//!   `tests/fullbattle_test.rs` (a per-seed PER-DECISION STATE+SEED differential to
//!   game-end: ~2000 per-decision EXACT seed assertions over 8 scenarios × 50 seeds
//!   — both-switch distinct/tie, switch-vs-move, post-faint single + double replace,
//!   KO-to-win, last-mon double-KO tie — INCLUDING the winner). DEFERRED: secondary
//!   effects, status MOVES, entry hazards (Spikes), Pursuit, Baton Pass, protocol
//!   emission.
//! - [`search`] — **DONE & validated**. The clone-and-branch SEARCH kernels
//!   (`gen3_rust_search_driver_v1`): the port of `utils/bridge/replay_kernels.js`'s
//!   search half (the aux PRNG, `random_choice`, `build_to_turn`, `resolve_turn` /
//!   `resolve_turn_exact`, and the omniscient `outcome_of` / `pre_state` renderers),
//!   composed by `src/bin/search_driver.rs` into a byte-compatible drop-in for
//!   `utils/bridge/search_driver.js`. Pinned by `tests/search_driver_test.rs` and, for
//!   the wire contract, by the golden differential `tmp/search_impl_parity.py`.
//! - [`protocol`], [`battle`] — **partial**. [`battle::Battle::start`] +
//!   `start_with_switchins` / `start_with_turn0_construction` +
//!   [`battle::BattleStream::write_line`] are real and validated; a handful of
//!   never-built convenience methods on [`battle::Battle`] (`new`/`choose`/`ended`/
//!   `winner`/`seed`/`reseed`) stay `todo!()` because every production path drives the
//!   engine through [`turn::FullBattleDriver`] instead. The clone-and-branch SEARCH
//!   snapshot is `Clone` (see [`bridge::BridgeSession::snapshot`]), not a byte format.
//!
//! ## Cross-gen note
//!
//! Gen 3 OU is the only target today, but the engine is kept **generation-generic**
//! (a generation parameter on [`dex::Dex`] + data-as-tables, mirroring Showdown's
//! gen9→gen3 mod-delta layering). Don't hard-code Gen-3 constants into the core;
//! put them behind the generation config so other gens slot in later without an
//! engine rewrite.

pub mod battle;
pub mod bridge;
pub mod damage;
pub mod dex;
pub mod event;
pub mod json;
pub mod prng;
pub mod protocol;
pub mod search;
pub mod state;
pub mod stats;
pub mod team;
pub mod turn;
