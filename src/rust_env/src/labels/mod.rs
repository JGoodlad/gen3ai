//! THE TRAINING LABELS — the label columns the env core writes per decision (M5 Lane C,
//! `designs/endstate/program_rust_core.md` §2 M5; the inventory and each key's derivation:
//! `designs/rust_sim/env_labels.md`).
//!
//! The columns and the family → column map are GENERATED (`core::columns::labels`, from
//! `src/utils/rust_env/label_inventory.py`). This module owns which families are BUILT and, per
//! family, the producer. A spec that declares a family the core does not write is refused at
//! STARTUP (the DECLARED LIFECYCLE): a `host_const` / `host_episode` family belongs to the host, a
//! `refused` one is off the production surface and not ported, a `core` family not yet built is
//! refused as such — never a silently stale column.

use crate::core::columns::labels::{FAMILIES, NOT_CORE};

/// The `core` families whose producer exists (grows one Lane-C unit at a time).
pub const BUILT: &[&str] = &[];

/// Validate the spec's `labels` declaration; the families in TABLE order (deterministic).
pub fn declare(names: &[String]) -> Result<Vec<&'static str>, String> {
    for (i, n) in names.iter().enumerate() {
        if names[..i].contains(n) {
            return Err(format!("spec: `labels` names {n:?} twice"));
        }
        if let Some((_, why)) = NOT_CORE.iter().find(|(f, _)| f == n) {
            return Err(match *why {
                "refused" => format!("spec: label family {n:?} is OFF the production surface and not ported to the Rust env (refused)"),
                _ => format!("spec: label family {n:?} is filled by the HOST ({why}), not by the core"),
            });
        }
        if !FAMILIES.iter().any(|(f, _)| f == n) {
            return Err(format!("spec: unknown label family {n:?}"));
        }
        if !BUILT.contains(&n.as_str()) {
            return Err(format!("spec: label family {n:?} is not built yet in the Rust env (M5 Lane C)"));
        }
    }
    Ok(FAMILIES.iter().map(|(f, _)| *f).filter(|f| names.iter().any(|n| n == f)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declaration_is_refused_by_kind() {
        assert_eq!(declare(&[]).unwrap(), Vec::<&str>::new());
        let e = |n: &str| declare(&[n.to_string()]).unwrap_err();
        assert!(e("dense_aux").contains("refused"), "{}", e("dense_aux"));
        assert!(e("winprob").contains("HOST"), "{}", e("winprob"));
        assert!(e("opp_class").contains("HOST"), "{}", e("opp_class"));
        assert!(e("nope").contains("unknown"));
        for (f, _) in FAMILIES {
            if !BUILT.contains(&f) {
                assert!(e(f).contains("not built"), "{f}");
            }
        }
    }
}
