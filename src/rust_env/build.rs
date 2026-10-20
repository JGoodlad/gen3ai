//! THE BUILD STAMP (M5 Lane 0, gate ⑤): what this build IS, so both front ends can REFUSE a stale or
//! foreign build at load (the 09-09 rust-target incident class — a binary from another checkout or
//! an older tree runs silently; a stale `.so` first on a search path is the same class).
//!
//! `pokesim_env::core::STAMP` =
//! `stamp=v2;kind=env;commit=<sha>;src=<fnv64>;nfiles=<k>;nan_poison=<0|1>;schema=<SCHEMA_ID>;data=<abs path>`:
//!
//! * `src` — FNV-1a-64 over the sorted `"<tag>/<rel>\t<git blob id>\n"` listing of every file that
//!   reaches a NON-TEST build of this crate and the port it path-depends on: the port's `src/**/*.rs`
//!   (tag `port`) and this crate's (tag `env`), each with its `Cargo.toml` and `build.rs` (and the port's
//!   `build_support.rs`), plus this crate's `Cargo.lock`. The file set and the algorithm live in
//!   `../rust_sim/build_support.rs`, shared with the port's own stamp; `src/utils/rust_env/stamp.py`
//!   recomputes it from the tree it imports from. THE refusal key.
//! * `nan_poison` — 1 when the port's rows are NaN-prefilled (a test / self-check build), 0 for a
//!   release build: the same source hash builds two binaries that write different bytes into an
//!   unwritten cell, so a caller can demand the one it means.
//! * `schema` — `columns::SCHEMA_ID`, read from the generated file (a front end compares it with its
//!   own table).
//! * `data` — the ABSOLUTE `data/pokemon` directory this build's port was compiled against, a
//!   fallback only: the directory a module READS is resolved at run time (`dex/data_dir.rs`).
//! * `commit` — informational (`rerun-if-changed` watches the sources, not `.git/HEAD`).

#[path = "../rust_sim/build_support.rs"]
mod build_support;

use std::path::PathBuf;

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let port = here.join("../rust_sim").canonicalize().expect("the port crate beside this one");
    let (src, n) = build_support::hash_scope(&[build_support::port_spec(&port, false), build_support::env_spec(&here)]);
    let commit = build_support::head_commit(&here);
    let nan_poison = u8::from(build_support::selfcheck_build());
    let columns = std::fs::read_to_string(here.join("src/core/columns.rs")).expect("the generated columns.rs");
    let schema = columns
        .lines()
        .find_map(|l| l.strip_prefix("pub const SCHEMA_ID: &str = \"").and_then(|r| r.strip_suffix("\";")))
        .expect("columns.rs declares SCHEMA_ID");
    let data = port.join("../../data/pokemon").canonicalize().expect("the data/pokemon directory the port reads");
    let data = data.to_string_lossy();
    assert!(!data.contains(';'), "the data path must not contain ';' (the stamp's separator)");
    println!(
        "cargo:rustc-env=POKESIM_ENV_STAMP=stamp=v2;kind=env;commit={commit};src={src};nfiles={n};nan_poison={nan_poison};schema={schema};data={data}"
    );
}
