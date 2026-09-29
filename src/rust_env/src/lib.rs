//! `pokesim_env` — the Rust env core (M5). See `core` for the contract; the column schema is
//! GENERATED (`core::columns`).
pub mod core;
/// FRONT END A — the C ABI over `core` (M5 Lane A; `src/utils/rust_env/ffi.py` loads it).
pub mod ffi;
/// The TRAINING LABELS (M5 Lane C): which label families are built, and their producers.
pub mod labels;
