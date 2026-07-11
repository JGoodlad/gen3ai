//! The HANDLER-COMPLETENESS AUDIT gate (`gen3_handler_audit_v1`) — the dispatch-bus's
//! completeness guarantee as a STATIC cargo gate.
//!
//! The port implements effects AT-SITE (no generic runEvent bus); the recurring bug
//! class is an effect carrying a handler at a hook we never enumerated (Immunity's
//! onUpdate cure, Cloud Nine's onEnd WeatherChange, Plus/Minus's cross-field
//! onModifySpA, facade's onBasePower, jumpkick's onMoveFail crash, the Freeze Clause
//! block). `harness/dump_gen3_handlers.js --audit` enumerates EVERY handler-bearing
//! key on EVERY effect in the reachable surface from the RESOLVED `Dex.mod('gen3')`
//! and fails when:
//!   (a) a resolved handler key has NO row in the committed manifest
//!       (`tests/vectors/gen3_handler_audit.json`) — a NEW/unnoticed handler;
//!   (b) a manifest row's key vanished from the dist — stale;
//!   (c) a body FINGERPRINT changed — the handler's semantics may have drifted
//!       (re-probe before re-accepting);
//!   (d) an `implemented` row's anchor no longer greps in src/ — the modeling
//!       site was renamed/removed.
//!
//! Requires the submodule dist/ + node_modules symlinks (the same requirement as the
//! golden GENERATORS documented in the root CLAUDE.md) — the test FAILS LOUDLY if
//! node or the dist is unavailable rather than silently skipping (a silently-skipped
//! completeness gate is no gate).

use std::path::Path;
use std::process::Command;

#[test]
fn handler_audit_gate() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new("node")
        .arg("harness/dump_gen3_handlers.js")
        .arg("--audit")
        .current_dir(dir)
        .output()
        .expect("node must be runnable for the handler-audit gate (see root CLAUDE.md worktree setup)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "HANDLER AUDIT FAILED — a new/stale/drifted/unanchored (effect, hook) row.\n\
         Triage it in harness/handler_audit_dispositions.js (probe first — the sim is the\n\
         only oracle), then regenerate the manifest with `node harness/dump_gen3_handlers.js`.\n\
         --- stderr ---\n{stderr}\n--- stdout ---\n{stdout}"
    );
    assert!(
        stderr.contains("handler audit OK"),
        "expected the audit OK banner, got:\n{stderr}"
    );
}
