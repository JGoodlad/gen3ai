//! THE PORT'S BUILD STAMP (`build.rs`): `stamp=v2;kind=port;commit=…;src=…;nfiles=…;selfcheck=…`.
//!
//! `src` hashes every file this crate compiles into a non-test build (`build_support.rs` defines the
//! set). Each binary prints it on `--build-stamp`; `utils.rust_env.stamp.check_binary_stamp` REFUSES a
//! binary whose `src` is not the checkout's. The env core's own stamp (`pokesim_env::core::STAMP`) covers
//! this crate too, so a library or child that links the port is judged by that one.

/// The stamp this build of the port carries.
pub const STAMP: &str = env!("POKESIM_PORT_STAMP");

#[cfg(test)]
mod tests {
    #[test]
    fn the_stamp_is_well_formed() {
        let f: Vec<(&str, &str)> = super::STAMP.split(';').map(|kv| kv.split_once('=').unwrap()).collect();
        let keys: Vec<&str> = f.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, ["stamp", "kind", "commit", "src", "nfiles", "selfcheck"]);
        assert_eq!(f[0], ("stamp", "v2"));
        assert_eq!(f[1], ("kind", "port"));
        assert_eq!(f[3].1.len(), 16, "src is 16 hex digits");
        assert!(f[3].1.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(f[4].1.parse::<usize>().unwrap() > 50, "the port's sources are in the listing");
        assert_eq!(f[5], ("selfcheck", if cfg!(any(debug_assertions, feature = "emission-selfcheck")) { "1" } else { "0" }));
    }
}
