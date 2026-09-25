//! `present()` — ONE side's reading of the battle, as poke-env holds it, in the Rust core
//! (`gen3_core_present_v1`, the Rust Core Program's milestone M2,
//! `designs/endstate/program_rust_core.md`; contract `designs/rust_sim/present.md`).
//!
//! # The model
//!
//! * [`BoardReading`] is poke-env's `Battle` + `Pokemon` for one side's stream: every transition the
//!   training observation's `LiveView` reads, each mirroring the poke-env method it is named
//!   after (`AbstractBattle.parse_message`, `Battle.parse_request`, `Battle.switch`,
//!   `Pokemon.moved` / `set_hp_status` / `start_effect` / …). It is fed TYPED lines
//!   ([`crate::core_events::Line`]) — from the source on the step path, from `Line::parse` of the
//!   text on the parse path — so the two paths fold the same values and cannot disagree.
//! * [`present`] is `LiveView.from_battle` over a board reading: a [`OneSidedView`] in `LiveView`'s own
//!   shape. Where poke-env's reading is WRONG about a sim fact the stream establishes, it carries
//!   the TRUTH — never a rule that reproduces the mistake; the disagreement is a named FINDING
//!   (`agents/battle/poke_env_findings.py`, `designs/rust_sim/present.md` §3).
//! * [`legal_actions`] is `LegalActions.from_battle` (the raw `|request|` + poke-env's parse of
//!   it) and [`mask`] the 11-dim action mask.
//! * The poke-env DATA the reading consults (its pokedex, move table, `Effect` lifecycle sets) is
//!   GENERATED into [`tables`] from poke-env itself (`python -m agents.battle.rust_core_present_tables`).
//!
//! # The reading rules — where `LiveView` is a PRESENTATION, not the simulator's state
//!
//! Named, applied at the named site, pinned by a unit test to the poke-env line it mirrors
//! (`tests` below, and `agents/battle/rust_core_present_test.py` against poke-env itself). The
//! ids continue `designs/rust_sim/one_sided_view.md` §2b's (V1–V13 are that table's rules; this
//! crate now applies all of them itself).
//!
//! | rule | the reading | poke-env | site |
//! |---|---|---|---|
//! | V1 | an opposing mon has a row once a line NAMED it (a switch shows it; `revealed` from the `\|switch\|`) | `get_pokemon`, `Pokemon.switch_in` | `BoardReading::get_pokemon`, `PMon::switch_in` |
//! | V2 | our slots in `battle.team` order (the first request's roster); theirs in first-named order | `_update_team_from_request`; dict insertion | `BoardReading::team` / `opp` |
//! | V3 | an opposing move's PP = max − uses, a use against a Pressure holder (judged with the ability poke-env held THEN) costing two | `Pokemon.moved` → `Move.use`; `_pressure_on` | `BoardReading::pressure_on`, `PMon::moved` |
//! | V4 | the volatile fold: start / count / end, `ends_on_turn` dropped at `\|turn\|`, cleared on switch-out and faint, Baton Pass copying `BATON_PASS_COPIED_EFFECTS` | `start_effect` / `end_effect` / `end_turn` / `_clear_effects` / `apply_baton_pass` | `PMon` |
//! | V5 | the sleep / toxic counter: +1 per `\|move\|` / `\|cant\|` while asleep; the toxic STAGE, +1 per residual `[from] psn` chip (cap 15), reset at the switch-in (fork PE-R1b fix); a CHANGED status restarts it (fork R1) | `moved` / `cant_move` / `note_residual_chip` / the `status` setter / `cure_status` / `switch_out` / `Battle.switch` | `PMon`, `BoardReading::switch` |
//! | V6 | the protect streak: a count of consecutive stall moves, not the sim's stall denominator | `moved` / `cant_move` / `switch_out` / `start_effect` | `PMon` |
//! | V7 | an item from `-item` / `-enditem` / the `_check_*_message_for_item` clauses only; `consumed_item` from `-enditem` | the four handlers | `BoardReading` |
//! | V8 | two ability slots (base + temporary), the single-possible-ability inference, Trace's double assignment, the four non-`-ability` disclosures | `Pokemon.ability` + setter, `_update_from_pokedex`, the `-ability` / `-activate` / `-immune` / `_check_*_for_ability` handlers | `PMon`, `BoardReading` |
//! | V9 | an opponent's spread / stats / exact HP are unknown | `LivePokemon.from_pokemon(is_own=False)` | `view::mon_view` |
//! | ~~V10~~ | a fainted mon holds NO stages (the sim's faint `clearVolatile`) — was FINDING PE-V10, FIXED in the fork (`gen3_pe_reading_fixes_v1`) | `faint()` clears boosts | `PMon::faint` |
//! | V11 | a timed screen is stored as the TURN it started, Spikes as its layers | `_side_start` | `BoardReading::side_start` |
//! | V12 | weather `turns_active` = now − the set turn; permanent iff an `[from] ability:` set | `Gen3Battle._update_weather` | `BoardReading::weather` |
//! | V13 | the legality flags are poke-env's parse of the request | `Battle.parse_request` | `legal` |
//! | V14 | a TRANSFORMED mon reads the target's dex types / base stats, its KNOWN ability and moves (at 5 PP) and its stages; our own copied ability (R4) is an INFORMATION LIMIT (no stream states it) | `Pokemon.transform` | `PMon::transform` |
//! | V15 | our BENCHED mons' PP is the sighting counter (only the active one is re-synced from the request, fork R3); a gen-3 request never states a benched mon's PP, an INFORMATION LIMIT (the audit checks the count can only lag) | `_sync_active_pp` | `BoardReading::sync_active_pp` |
//! | V16 | using a damaging Electric move ends Charge (as the sim's `onAfterMove` does). Flash Fire is NOT ended by a Fire move (the sim's `flashfire` lasts until the holder leaves the field) — was FINDING PE-V16, FIXED in the fork (`gen3_pe_reading_fixes_v1`) | `Pokemon.moved` (silent effect ending) | `PMon::moved` |
//! | V17 | a request that says a mon is not active re-applies its last record and switches it out (`was_illusioned`); one that says it is, switches it in | `_update_team_from_request` | `BoardReading::update_team_from_request` |
//! | ~~R1b~~ | the toxic count is the STAGE (the residual `[from] psn` chips since its switch-in) — was FINDING PE-R1b (upstream ticked per `\|turn\|`), FIXED in the fork (`gen3_pe_reading_fixes_v1`); now part of V5 | `Pokemon.note_residual_chip` | `PMon::note_residual_chip` |

pub mod audit;
pub mod dex;
pub mod legal;
pub mod mon;
pub mod tables;
pub mod board_reading;
pub mod tokens;
pub mod view;

pub use legal::{legal_actions, mask, LegalActions};
pub use tokens::{choice_tokens, tokens_json};
pub use board_reading::BoardReading;
pub use audit::{check_view, Audit};
pub use view::{present, OneSidedView};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod called_move_tests;
