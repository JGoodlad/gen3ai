//! THE BUILD STAMP (M5 Lane 0, gate ⑤): what this build IS, so both front ends can REFUSE a stale or
//! foreign build at load (the 09-09 rust-target incident class — a binary from another checkout or
//! an older tree runs silently; a stale `.so` first on a search path is the same class).
//!
//! `pokesim_env::core::STAMP` =
//! `stamp=v1;commit=<sha>;src=<fnv64>;nfiles=<k>;nan_poison=<0|1>;schema=<SCHEMA_ID>;data=<abs path>`:
//!
//! * `src` — FNV-1a-64 over the sorted `"<tag>/<rel>\t<git blob id>\n"` listing of every file that
//!   reaches the build: the port's `src/**/*.rs` + `Cargo.toml` (tag `port`) and this crate's
//!   `src/**/*.rs` + `Cargo.toml` + `build.rs` (tag `env`). `src/utils/rust_env/stamp.py` recomputes
//!   it from the tree it imports from — keep the two algorithms identical. THE refusal key.
//! * `nan_poison` — 1 when the port's rows are NaN-prefilled (a test / self-check build), 0 for a
//!   release build: the same source hash builds two binaries that write different bytes into an
//!   unwritten cell, so a caller can demand the one it means.
//! * `schema` — `columns::SCHEMA_ID`, read from the generated file (a front end compares it with its
//!   own table).
//! * `data` — the ABSOLUTE `data/pokemon` directory this build reads at run time (the port's
//!   compile-time path): a build from another checkout reads ANOTHER checkout's data.
//! * `commit` — informational (`rerun-if-changed` watches the sources, not `.git/HEAD`).

use std::path::{Path, PathBuf};

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

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let port = here.join("../rust_sim").canonicalize().expect("the port crate beside this one");
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for (root, tag) in [(&port, "port"), (&here, "env")] {
        let mut v = Vec::new();
        walk(&root.join("src"), &mut v);
        v.push(root.join("Cargo.toml"));
        if tag == "env" {
            v.push(root.join("build.rs"));
        }
        for f in v {
            let rel = f.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            files.push((format!("{tag}/{rel}"), f));
        }
    }
    files.sort();
    // A directory watch is recursive: a source ADDED or removed re-runs the stamp too.
    println!("cargo:rerun-if-changed={}", port.join("src").display());
    println!("cargo:rerun-if-changed={}", here.join("src").display());
    let mut child = std::process::Command::new("git")
        .args(["hash-object", "--no-filters", "--stdin-paths"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("git hash-object");
    {
        use std::io::Write;
        let mut sin = child.stdin.take().unwrap();
        for (_, f) in &files {
            writeln!(sin, "{}", f.display()).unwrap();
        }
    }
    let out = child.wait_with_output().unwrap();
    let blobs: Vec<String> = String::from_utf8(out.stdout).unwrap().lines().map(str::to_string).collect();
    assert_eq!(blobs.len(), files.len(), "git hash-object returned the wrong count");
    let mut listing = String::new();
    for ((rel, f), b) in files.iter().zip(&blobs) {
        listing.push_str(rel);
        listing.push('\t');
        listing.push_str(b);
        listing.push('\n');
        println!("cargo:rerun-if-changed={}", f.display());
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in listing.as_bytes() {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&here)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let nan_poison = std::env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some()
        || std::env::var_os("CARGO_FEATURE_EMISSION_SELFCHECK").is_some();
    let columns = std::fs::read_to_string(here.join("src/core/columns.rs")).expect("the generated columns.rs");
    let schema = columns
        .lines()
        .find_map(|l| l.strip_prefix("pub const SCHEMA_ID: &str = \"").and_then(|r| r.strip_suffix("\";")))
        .expect("columns.rs declares SCHEMA_ID");
    let data = port.join("../../data/pokemon").canonicalize().expect("the data/pokemon directory the port reads");
    let data = data.to_string_lossy();
    assert!(!data.contains(';'), "the data path must not contain ';' (the stamp's separator)");
    println!(
        "cargo:rustc-env=POKESIM_ENV_STAMP=stamp=v1;commit={commit};src={h:016x};nfiles={};nan_poison={};schema={schema};data={data}",
        files.len(),
        u8::from(nan_poison)
    );
}
