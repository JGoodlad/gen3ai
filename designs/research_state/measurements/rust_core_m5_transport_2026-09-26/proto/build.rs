//! The BUILD STAMP: the git commit the build ran at + an FNV-1a-64 hash over every source file
//! that reaches the binary (the port's `src/` and this crate's `src/`). Python recomputes the same
//! hash from the tree it imports from and REFUSES a mismatch — the 09-09 rust-target incident class
//! (a stale or foreign `.so` first on the search path runs silently).
//! `m5_loader.source_hash` is the Python twin; keep the two algorithms identical.
//! The commit half is informational: `rerun-if-changed` watches the sources, not `.git/HEAD`, so a
//! commit that touches no source keeps the old commit string — the SOURCE hash is the refusal key.

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
    let port = here.join("../../../../../src/rust_sim").canonicalize().unwrap();
    let mut files = Vec::new();
    for (root, tag) in [(port.join("src"), "port"), (here.join("src"), "proto")] {
        let mut v = Vec::new();
        walk(&root, &mut v);
        for f in v {
            let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
            files.push((format!("{tag}/{rel}"), f));
        }
    }
    files.sort();
    // Per-file git BLOB ids (what `git hash-object` gives; Python's hashlib.sha1 reproduces them in
    // C), then FNV-1a-64 over the small "rel\tblob\n" listing — cheap to recompute at import.
    let mut child = std::process::Command::new("git")
        .args(["hash-object", "--stdin-paths"])
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
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=M5_STAMP_COMMIT={commit}");
    println!("cargo:rustc-env=M5_STAMP_SRC={h:016x}");
    println!("cargo:rustc-env=M5_STAMP_NFILES={}", files.len());
}
