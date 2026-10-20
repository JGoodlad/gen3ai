//! The build stamp's shared helper (`build_support.rs`, the half `rust_sim/build.rs` and
//! `rust_env/build.rs` both run): SHA-1 / git blob ids against known vectors, the test-only file rule,
//! and the listing over a real temporary tree (what is hashed, what is not, and that one byte flips it).
//! The cross-language agreement with `src/utils/rust_env/stamp.py` is `stamp_test.py`'s.

#[path = "../build_support.rs"]
mod build_support;

use build_support::*;
use std::path::PathBuf;

#[test]
fn sha1_and_git_blob_ids_match_known_vectors() {
    let hex = |d: [u8; 20]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert_eq!(hex(sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(hex(sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    // 56 bytes: the padding spills into a second block
    assert_eq!(hex(sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")), "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    // `printf 'hello\n' | git hash-object --stdin` and the empty blob
    assert_eq!(git_blob_id(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a");
    assert_eq!(git_blob_id(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
    assert_eq!(format!("{:016x}", fnv64(b"")), "cbf29ce484222325");
    assert_eq!(format!("{:016x}", fnv64(b"a")), "af63dc4c8601ec8c");
}

#[test]
fn test_only_files_are_excluded_by_name_and_nothing_else() {
    for rel in ["src/present/tests.rs", "src/encoder/tests.rs", "src/present/called_move_tests.rs",
                "src/x/foo_test.rs", "src/tests/a.rs", "tests/regression_test.rs"] {
        assert!(is_test_only(rel), "{rel} is test-only");
    }
    for rel in ["src/present/mod.rs", "src/turn.rs", "src/bin/sim_bridge.rs", "src/protocol.rs", "src/latest.rs",
                "src/contest.rs", "src/dex/test_support.rs", "Cargo.toml", "build.rs"] {
        assert!(!is_test_only(rel), "{rel} reaches the build");
    }
}

fn tree(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("pokesim_build_stamp_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src/sub")).unwrap();
    std::fs::create_dir_all(root.join("tests")).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[package]\n").unwrap();
    std::fs::write(root.join("build.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("build_support.rs"), "// helper\n").unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub mod sub;\n").unwrap();
    std::fs::write(root.join("src/sub/mod.rs"), "pub fn f() {}\n").unwrap();
    std::fs::write(root.join("src/sub/tests.rs"), "// test-only\n").unwrap();
    std::fs::write(root.join("tests/it.rs"), "// integration test\n").unwrap();
    std::fs::write(root.join("README.md"), "docs\n").unwrap();
    root
}

fn rels(root: &PathBuf, lock: bool) -> Vec<String> {
    let mut v: Vec<String> = crate_files(&port_spec(root, lock)).into_iter().map(|(r, _)| r).collect();
    v.sort();
    v
}

#[test]
fn the_listing_holds_what_a_non_test_build_compiles() {
    let root = tree("listing");
    assert_eq!(rels(&root, true), ["port/Cargo.toml", "port/build.rs", "port/build_support.rs", "port/src/lib.rs", "port/src/sub/mod.rs"]);
    std::fs::write(root.join("Cargo.lock"), "# lock\n").unwrap();
    assert!(rels(&root, true).contains(&"port/Cargo.lock".to_string()), "the root crate's lock is hashed");
    assert!(!rels(&root, false).contains(&"port/Cargo.lock".to_string()), "a non-root crate's lock is not");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn one_source_byte_flips_the_hash_and_a_test_file_does_not() {
    let root = tree("hash");
    let h = |lock| listing_text(crate_files(&port_spec(&root, lock)));
    let (base, n) = h(true);
    assert_eq!(n, 5);
    std::fs::write(root.join("src/sub/tests.rs"), "// edited test-only\n").unwrap();
    std::fs::write(root.join("tests/it.rs"), "// edited\n").unwrap();
    std::fs::write(root.join("README.md"), "edited docs\n").unwrap();
    assert_eq!(h(true).0, base, "test-only files, tests/ and docs are not hashed");
    std::fs::write(root.join("src/sub/mod.rs"), "pub fn f() {} \n").unwrap();
    assert_ne!(h(true).0, base, "one source byte changes the listing");
    let _ = std::fs::remove_dir_all(&root);
}
