//! `pokesim_env` — the Rust env core (M5). See `core` for the contract; the column schema is
//! GENERATED (`core::columns`).
pub mod core;
/// EPISODES AND REWARD (M5 Lane D): the terminal reward, `terminated` / `truncated`, the stall
/// forfeit, ties, the (absent) terminal observation, and the refused-start PARK.
pub mod episode;
/// FRONT END A — the C ABI over `core` (M5 Lane A; `src/utils/rust_env/ffi.py` loads it).
pub mod ffi;
/// The TRAINING LABELS (M5 Lane C): which label families are built, and their producers.
pub mod labels;
/// FRONT END B's transport — the shared mapping of the process front end (M5 Lane B; the child is
/// `src/bin/rust_env_proc.rs`, the host `src/utils/rust_env/proc.py`).
pub mod shm;
