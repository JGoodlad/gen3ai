//! THE PORT'S BUILD STAMP: what this build of the `pokesim` crate IS, so Python can REFUSE a binary
//! (`sim_bridge`, `search_driver`, `core_events`, `live_reader`, ...) whose sources are not the
//! checkout that is about to spawn it. A binary the resolver takes by path (`$POKESIM_*_BIN`, a
//! pre-built `target/`) otherwise runs behaviour no commit pins.
//!
//! `pokesim::stamp::STAMP` =
//! `stamp=v2;kind=port;commit=<sha>;src=<fnv64>;nfiles=<k>;selfcheck=<0|1>`; every binary prints it on
//! `--build-stamp` (`dex::data_dir::exit_on_flags`), and `utils.rust_env.stamp.check_binary_stamp`
//! compares `src` / `nfiles` with a recompute over the checkout it runs in.
//!
//! The hash and its file set are defined ONCE, in `build_support.rs` (shared with `rust_env/build.rs`
//! and mirrored by `src/utils/rust_env/stamp.py`).

#[path = "build_support.rs"]
mod build_support;

use std::path::PathBuf;

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let (src, n) = build_support::hash_scope(&[build_support::port_spec(&here, true)]);
    let commit = build_support::head_commit(&here);
    let selfcheck = u8::from(build_support::selfcheck_build());
    println!("cargo:rustc-env=POKESIM_PORT_STAMP=stamp=v2;kind=port;commit={commit};src={src};nfiles={n};selfcheck={selfcheck}");
}
