//! WHERE the static data lives — resolved at RUN time (`gen3_dex_runtime_data_dir_v1`).
//!
//! The 2026-09-09 incident: the data path was baked at COMPILE time (`env!("CARGO_MANIFEST_DIR")`),
//! so a binary built in a worktree but sitting in main's `target/` read the WORKTREE's
//! `data/pokemon`, and panicked once the worktree was deleted. The data a binary reads now follows
//! where the binary IS, never where it was built:
//!
//! 1. `$POKESIM_DATA_DIR` — authoritative when set (a missing file then fails at load, naming the
//!    path; it never falls through to a guess);
//! 2. the repo that holds the running MODULE — the shared object this code was loaded from (the
//!    env core's cdylib inside Python), else the executable — found by walking its ancestors for
//!    `data/pokemon/gen3_species.json`;
//! 3. the compile-time default (`<crate>/../../data/pokemon`) — only for a module outside any
//!    checkout.
//!
//! Every binary prints the resolution on `--version` ([`exit_on_version_flag`]), and its build stamp on
//! `--build-stamp`.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The run-time override (an absolute or cwd-relative directory holding the `gen3_*.json` files).
pub const DATA_DIR_ENV: &str = "POKESIM_DATA_DIR";
/// The file whose presence marks `<dir>/data/pokemon` as a repo's data directory.
const MARKER: &str = "gen3_species.json";

/// Where a resolved data directory came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataDirSource {
    /// `$POKESIM_DATA_DIR`.
    Env,
    /// The repo that holds this module (the path is the module file).
    Module(PathBuf),
    /// The compile-time default (no env var, no repo above the module).
    CompileTime,
}

impl DataDirSource {
    pub fn describe(&self) -> String {
        match self {
            DataDirSource::Env => format!("env:{DATA_DIR_ENV}"),
            DataDirSource::Module(p) => format!("module:{}", p.display()),
            DataDirSource::CompileTime => "compile-time".to_string(),
        }
    }
}

/// The compile-time default: `<repo the crate was BUILT in>/data/pokemon` (the last resort).
pub fn compile_time_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/pokemon")
}

/// THE RESOLUTION RULE, pure (no process state): `env` → the first ancestor of `module` holding
/// `data/pokemon/gen3_species.json` → `compile_default`. An EMPTY env value counts as unset.
pub fn resolve_data_dir(env: Option<&OsStr>, module: Option<&Path>, compile_default: &Path) -> (PathBuf, DataDirSource) {
    if let Some(v) = env.filter(|v| !v.is_empty()) {
        return (PathBuf::from(v), DataDirSource::Env);
    }
    if let Some(m) = module {
        for dir in m.ancestors().skip(1) {
            let cand = dir.join("data").join("pokemon");
            if cand.join(MARKER).is_file() {
                return (cand, DataDirSource::Module(m.to_path_buf()));
            }
        }
    }
    (compile_default.to_path_buf(), DataDirSource::CompileTime)
}

/// The file a code address was loaded from, read from a `/proc/self/maps` listing (Linux):
/// the mapping whose range holds `addr`, its path with any ` (deleted)` suffix dropped (a binary
/// replaced by a rebuild while it runs still names the directory it lives in).
pub fn module_in_maps(maps: &str, addr: usize) -> Option<PathBuf> {
    for line in maps.lines() {
        let mut it = line.splitn(6, ' ');
        let (Some(range), _, _, _, _, Some(rest)) = (it.next(), it.next(), it.next(), it.next(), it.next(), it.next()) else {
            continue;
        };
        let Some((lo, hi)) = range.split_once('-') else { continue };
        let (Ok(lo), Ok(hi)) = (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16)) else { continue };
        if addr < lo || addr >= hi {
            continue;
        }
        let path = rest.trim_start();
        let path = path.strip_suffix(" (deleted)").unwrap_or(path);
        return path.starts_with('/').then(|| PathBuf::from(path));
    }
    None
}

/// The module this code runs from: the mapping of this function (a cdylib inside a host process
/// names the `.so`, an executable names itself), else `current_exe()`.
fn running_module() -> Option<PathBuf> {
    let addr = running_module as fn() -> Option<PathBuf> as usize;
    std::fs::read_to_string("/proc/self/maps")
        .ok()
        .and_then(|m| module_in_maps(&m, addr))
        .or_else(|| std::env::current_exe().ok())
}

/// The process's data directory and where it came from — resolved ONCE, at first use.
pub fn data_dir_resolution() -> &'static (PathBuf, DataDirSource) {
    static R: OnceLock<(PathBuf, DataDirSource)> = OnceLock::new();
    R.get_or_init(|| {
        let env = std::env::var_os(DATA_DIR_ENV);
        resolve_data_dir(env.as_deref(), running_module().as_deref(), &compile_time_data_dir())
    })
}

/// The process's data directory (`<repo>/data/pokemon`).
pub fn data_dir() -> PathBuf {
    data_dir_resolution().0.clone()
}

/// The `--version` line: the binary, the resolved data directory, its source, and the
/// compile-time default (so a foreign build is visible at a glance).
pub fn version_line(bin: &str) -> String {
    let (dir, src) = data_dir_resolution();
    format!(
        "{bin} pokesim {}\tdata_dir={}\tsource={}\tcompile_default={}",
        env!("CARGO_PKG_VERSION"),
        dir.display(),
        src.describe(),
        compile_time_data_dir().display()
    )
}

/// `<bin> --version` / `<bin> --build-stamp` (as the FIRST argument): print [`version_line`], or this
/// crate's build stamp ([`crate::stamp::STAMP`]), and exit 0. Call it first in every port binary's `main`.
pub fn exit_on_version_flag(bin: &str) {
    exit_on_flags(bin, crate::stamp::STAMP);
}

/// The first-argument flags every binary answers: `--version` ([`version_line`]) and `--build-stamp`
/// (`build_stamp`, the BINARY'S OWN crate's stamp: a `rust_env` binary passes `pokesim_env::core::STAMP`,
/// which covers the port too). The Python resolvers run `--build-stamp` before they spawn a binary.
pub fn exit_on_flags(bin: &str, build_stamp: &str) {
    match std::env::args().nth(1).as_deref() {
        Some("--version") => {
            println!("{}", version_line(bin));
            std::process::exit(0);
        }
        Some("--build-stamp") => {
            println!("{build_stamp}");
            std::process::exit(0);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_repo(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("pokesim_data_dir_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("data/pokemon")).unwrap();
        std::fs::write(root.join("data/pokemon").join(MARKER), "{}").unwrap();
        std::fs::create_dir_all(root.join("src/rust_sim/target/release")).unwrap();
        root
    }

    #[test]
    fn the_module_location_wins_over_the_build_location() {
        // THE INCIDENT: a binary in repo A's target/, compiled in (now deleted) repo B.
        let a = fake_repo("module");
        let exe = a.join("src/rust_sim/target/release/sim_bridge");
        let b = Path::new("/nonexistent/gen3ai-wt-deleted/data/pokemon");
        let (dir, src) = resolve_data_dir(None, Some(&exe), b);
        assert_eq!(dir, a.join("data/pokemon"));
        assert_eq!(src, DataDirSource::Module(exe.clone()));
        let _ = std::fs::remove_dir_all(&a);
    }

    #[test]
    fn the_env_var_is_authoritative_and_empty_is_unset() {
        let a = fake_repo("env");
        let exe = a.join("src/rust_sim/target/release/sim_bridge");
        let d = Path::new("/compile/default");
        let (dir, src) = resolve_data_dir(Some(OsStr::new("/elsewhere/pokemon")), Some(&exe), d);
        assert_eq!((dir, src), (PathBuf::from("/elsewhere/pokemon"), DataDirSource::Env));
        let (dir, _) = resolve_data_dir(Some(OsStr::new("")), Some(&exe), d);
        assert_eq!(dir, a.join("data/pokemon"));
        let _ = std::fs::remove_dir_all(&a);
    }

    #[test]
    fn a_module_outside_any_repo_falls_back_to_the_compile_time_default() {
        let d = Path::new("/compile/default");
        let (dir, src) = resolve_data_dir(None, Some(Path::new("/nonexistent/bin/x")), d);
        assert_eq!((dir, src), (d.to_path_buf(), DataDirSource::CompileTime));
        let (dir, src) = resolve_data_dir(None, None, d);
        assert_eq!((dir, src), (d.to_path_buf(), DataDirSource::CompileTime));
    }

    #[test]
    fn maps_lookup_finds_the_mapping_and_drops_deleted() {
        let maps = "\
55d0a0000000-55d0a0100000 r-xp 00000000 08:02 123 /repo/src/rust_env/target/release/libpokesim_env.so (deleted)
7f0000000000-7f0000001000 rw-p 00000000 00:00 0
7f0000001000-7f0000002000 r-xp 00000000 00:00 0                          [vdso]
";
        assert_eq!(
            module_in_maps(maps, 0x55d0a0000010),
            Some(PathBuf::from("/repo/src/rust_env/target/release/libpokesim_env.so"))
        );
        assert_eq!(module_in_maps(maps, 0x7f0000000010), None);
        assert_eq!(module_in_maps(maps, 0x7f0000001010), None);
        assert_eq!(module_in_maps(maps, 0x10), None);
    }

    #[test]
    fn this_test_binary_resolves_this_checkouts_data() {
        // `cargo test` runs from `<checkout>/src/rust_sim/target/<profile>/deps/` (or a
        // CARGO_TARGET_DIR elsewhere — then the compile-time default, the same checkout).
        if std::env::var_os(DATA_DIR_ENV).is_some() {
            return;
        }
        let want = compile_time_data_dir().canonicalize().unwrap();
        assert_eq!(data_dir().canonicalize().unwrap(), want);
    }
}
