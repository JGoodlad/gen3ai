//! THE BUILD STAMP'S SHARED HALF — included by `rust_sim/build.rs` and (through `#[path]`) by
//! `rust_env/build.rs`, and mirrored by `src/utils/rust_env/stamp.py`. Keep the three identical: the
//! cross-language agreement is proven by `stamp_test.py` (the algorithm) and by every checked build
//! (`ffi.load`, `proc` handshake, `--build-stamp` of each driver binary).
//!
//! WHAT THE HASH IS. For each crate in the build's SCOPE (`port` = this crate alone, what `sim_bridge`,
//! `search_driver`, `core_events` and `live_reader` compile; `env` = the port plus `rust_env`, what the
//! env core's cdylib / `rust_env_proc` / `bot_reader` compile) the listing holds one line per file
//! that reaches a NON-TEST build:
//!
//! * every `src/**/*.rs` that is not TEST-ONLY ([`is_test_only`]);
//! * `Cargo.toml`, `build.rs`, and (port only) this file;
//! * `Cargo.lock` of the crate the build is ROOTED in, if present. It is gitignored and written by
//!   cargo itself, and cargo writes only the root crate's, so the port's lock is NOT part of the
//!   `env` scope (a later port build would create it and flip the env hash).
//!
//! and not `tests/`, `examples/`, `harness/`, goldens or docs. A line is `"<tag>/<rel>\t<git blob id>\n"`;
//! the listing is sorted, and its FNV-1a-64 is the stamp's `src`. The blob id is SHA-1 of
//! `"blob <len>\0" + bytes` — what `git hash-object` prints, computed here so the build needs no `git`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One crate in a build's scope.
pub struct CrateSpec {
    /// The listing prefix: `port` or `env`.
    pub tag: &'static str,
    /// The crate's directory (holds `Cargo.toml` and `src/`).
    pub root: PathBuf,
    /// Files beside `Cargo.toml` that the build runs or includes.
    pub extras: &'static [&'static str],
    /// Whether this crate's `Cargo.lock` is part of the listing (the build's ROOT crate only).
    pub lock: bool,
}

/// The port crate (`src/rust_sim`). `lock` is true when it is the build's root.
pub fn port_spec(root: &Path, lock: bool) -> CrateSpec {
    CrateSpec { tag: "port", root: root.to_path_buf(), extras: &["build.rs", "build_support.rs"], lock }
}

/// The env crate (`src/rust_env`); always the root of its own builds.
pub fn env_spec(root: &Path) -> CrateSpec {
    CrateSpec { tag: "env", root: root.to_path_buf(), extras: &["build.rs"], lock: true }
}

/// A source file that only a TEST build compiles (`#[cfg(test)] mod x;`), by NAME: a file called
/// `tests.rs`, `*_tests.rs` or `*_test.rs`, or anything under a `tests/` directory. `rel` is
/// crate-relative with `/` separators. `stamp_test.py` cross-checks this against the tree's actual
/// `#[cfg(test)] mod` declarations, so a test module that breaks the convention (or a non-test file
/// that borrows a test name) fails a gate instead of silently changing what the stamp covers.
/// An inline `#[cfg(test)] mod tests { .. }` block inside a source file is part of that file's hash
/// (an edit to it demands a rebuild: the conservative direction).
pub fn is_test_only(rel: &str) -> bool {
    let mut parts = rel.split('/').peekable();
    while let Some(p) = parts.next() {
        if parts.peek().is_some() {
            if p == "tests" {
                return true;
            }
        } else {
            let stem = p.strip_suffix(".rs").unwrap_or(p);
            return stem == "tests" || stem.ends_with("_tests") || stem.ends_with("_test");
        }
    }
    false
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `(tag/rel, path)` of every file of `c` that reaches a non-test build (unsorted).
pub fn crate_files(c: &CrateSpec) -> Vec<(String, PathBuf)> {
    let mut srcs = Vec::new();
    walk(&c.root.join("src"), &mut srcs);
    let mut paths: Vec<PathBuf> = srcs;
    paths.push(c.root.join("Cargo.toml"));
    for e in c.extras {
        paths.push(c.root.join(e));
    }
    if c.lock {
        paths.push(c.root.join("Cargo.lock"));
    }
    let mut out = Vec::new();
    for p in paths {
        if !p.is_file() {
            continue;
        }
        let rel = p.strip_prefix(&c.root).unwrap().to_string_lossy().replace('\\', "/");
        if is_test_only(&rel) {
            continue;
        }
        out.push((format!("{}/{}", c.tag, rel), p));
    }
    out
}

/// SHA-1 (FIPS 180-1), std-only; used for git blob ids only (not for security).
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// `git hash-object --no-filters` of `data`, as 40 hex digits.
pub fn git_blob_id(data: &[u8]) -> String {
    let mut buf = format!("blob {}\0", data.len()).into_bytes();
    buf.extend_from_slice(data);
    sha1(&buf).iter().map(|b| format!("{b:02x}")).collect()
}

/// FNV-1a-64 (the stamp's listing hash).
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in bytes {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The sorted `"<tag>/<rel>\t<blob id>\n"` listing of `files` and the number of files in it.
pub fn listing_text(mut files: Vec<(String, PathBuf)>) -> (String, usize) {
    files.sort();
    let mut text = String::new();
    for (rel, p) in &files {
        let data = std::fs::read(p).unwrap_or_else(|e| panic!("build stamp: cannot read {}: {e}", p.display()));
        text.push_str(rel);
        text.push('\t');
        text.push_str(&git_blob_id(&data));
        text.push('\n');
    }
    (text, files.len())
}

/// The stamp's `src` and `nfiles` over `scope` (crates in order, root last), and the cargo
/// `rerun-if-changed` lines that make a build script re-run when any listed file changes or a
/// source is added (a directory watch is recursive).
pub fn hash_scope(scope: &[CrateSpec]) -> (String, usize) {
    let mut files = Vec::new();
    for c in scope {
        println!("cargo:rerun-if-changed={}", c.root.join("src").display());
        files.extend(crate_files(c));
    }
    for (_, p) in &files {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    let (text, n) = listing_text(files);
    (format!("{:016x}", fnv64(text.as_bytes())), n)
}

/// `git rev-parse HEAD` of the checkout holding `dir`, or `unknown`. INFORMATIONAL only (cargo does
/// not re-run a build script when `.git/HEAD` moves), so it is never a refusal key.
pub fn head_commit(dir: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Whether the emission self-check (and NaN-prefilled rows) is compiled in: a `debug_assertions`
/// build or the `emission-selfcheck` feature. Read from the build script's cargo environment.
pub fn selfcheck_build() -> bool {
    std::env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some() || std::env::var_os("CARGO_FEATURE_EMISSION_SELFCHECK").is_some()
}
