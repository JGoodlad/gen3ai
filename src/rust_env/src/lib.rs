//! `pokesim_env` — the Rust env core (M5). See `core` for the contract; the column schema is
//! GENERATED (`core::columns`).
pub mod core;
/// A SCRIPTED BOT OVER ONE SIDE'S PROTOCOL STREAM (poke-env retirement P6): the Lane-F bot on
/// `SideReader`'s reading, for a websocket client (`bin/bot_reader.rs`).
pub mod bot_side;
/// THE SCRIPTED BOTS (M5 Lane F): the inventoried bots, reading the core's per-side reading.
pub mod bots;
/// EPISODES AND REWARD (M5 Lane D): the terminal reward, `terminated` / `truncated`, the stall
/// forfeit, ties, the (absent) terminal observation, and the refused-start PARK.
pub mod episode;
/// FRONT END A — the C ABI over `core` (M5 Lane A; `src/utils/rust_env/ffi.py` loads it).
pub mod ffi;
/// The TRAINING LABELS (M5 Lane C): which label families are built, and their producers.
pub mod labels;
/// OPPONENT ROUTING (M5 Lane E): the per-episode route table (external / T2 policy slot / Lane F bot).
pub mod opponents;
/// SEARCH ON `successors()` IN PROCESS (M5 Lane I): the search tree and the playouts.
pub mod search;
/// FRONT END B's transport — the shared mapping of the process front end (M5 Lane B; the child is
/// `src/bin/rust_env_proc.rs`, the host `src/utils/rust_env/proc.py`).
pub mod shm;
/// THE PERSISTED TRACE of a finished episode (M5 Lane H): its `gen3_core_event_v1` records and
/// reconstruction record, replayed from its input log.
pub mod trace;
