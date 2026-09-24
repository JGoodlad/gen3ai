//! The EMISSION SELF-CHECK (`gen3_core_emission_selfcheck_v1`): every protocol line the port
//! emits is checked AT THE MOMENT it is emitted, not afterwards by a whole-battle harness.
//!
//! # What is asserted
//!
//! 1. **The omniscient line round-trips exactly** — at every [`crate::protocol::ProtocolBuilder`]
//!    commit (and every `attrLastMove` retro-edit): `Line::parse(render(l)) == l`
//!    ([`check_omniscient`]).
//! 2. **Each viewer's render is the line that viewer is supposed to see** — at every per-side
//!    derivation (`bridge::derive_side`, the one funnel both the framing and every flushed batch
//!    go through): the viewer receives the line iff it is not owner-only for the other side, and
//!    the text it receives parses back to `side_view(l, viewer)` — the TYPED privacy fold, an
//!    implementation independent of the text fold being checked ([`check_side`]).
//! 3. **The `|split|` form** — `bridge::split_log_lines` (the replay family's `battle.log` shape):
//!    the secret half is the owner's line, the shared half the other viewer's (EMPTY for an
//!    owner-only line), and an un-split line reads the same to both viewers ([`check_split`],
//!    [`check_broadcast`]).
//! 4. **The bridge's own frames** (a `|request|`, an `|error|`, the forced-Struggle announce): the
//!    text parses losslessly under a known keyword, and a frame that belongs to one side names
//!    that side ([`check_frame`]).
//!
//! **A secret field never reaches the other viewer**: every per-viewer check also asserts, as a
//! separate and spec-level invariant, that the owner's EXACT HP (in a percent format) and an
//! owner-only line are absent from the non-owner's render — so a defect shared by BOTH folds
//! (text and typed) is still caught.
//!
//! # When it runs — and why production cannot pay for it
//!
//! [`ENABLED`] is `cfg!(any(debug_assertions, feature = "emission-selfcheck"))`, and every CALL
//! SITE is behind `#[cfg(any(debug_assertions, feature = "emission-selfcheck"))]`, so in a build
//! with neither the calls are not compiled at all (the checks below are still compiled and unit
//! tested, so they cannot rot between the builds that use them):
//!
//! * `cargo test` (the `test` profile has `debug_assertions`) — ON, every test, every binary it
//!   spawns.
//! * the fuzzers, the slice E/V corpora and the Python fuzz scripts — ON, through the SELF-CHECK
//!   build: `cargo build --profile selfcheck --features emission-selfcheck` writes to
//!   `target/selfcheck/`, a directory the production resolver never reads, so a self-check build
//!   can never overwrite the binary a live run execs.
//! * `cargo build --release` (what `utils.bridge.sim_bridge_bin` builds for training) — OFF,
//!   compiled out.
//!
//! # On failure
//!
//! It PANICS with [`MARKER`], the line, both viewers' renders and the viewer. A test build never
//! continues past one: the binaries that catch panics per request (`sim_bridge`,
//! `search_driver`) exit the process on a self-check failure ([`exit_if_failure`]) instead of
//! turning it into a recoverable `__ERR__`.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::core_events::line::{Field, Hp, Line};
use crate::core_events::schema::Kw;
use crate::core_events::side::side_view;

/// How many checks of each kind this process ran — `[omniscient, per_viewer, split, frame]`
/// (relaxed; a count, never a synchronization point). What lets a harness PROVE the binary it
/// ran had the check on: a production build never calls a check, so its counts stay zero.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn bump(i: usize) {
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
}

/// The per-kind check counts so far: `[omniscient, per_viewer, split, frame]`.
pub fn counts() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| COUNTS[i].load(Ordering::Relaxed))
}

/// The one-line summary a binary prints to STDERR at exit in a self-check build
/// (`core_events`; `agents.battle.rust_core_parity` reads it into its census).
pub fn summary() -> String {
    let [o, v, s, f] = counts();
    format!("emission_selfcheck omniscient={o} per_viewer={v} split={s} frame={f}")
}

/// Whether this build runs the emission self-check.
pub const ENABLED: bool = cfg!(any(debug_assertions, feature = "emission-selfcheck"));

/// The prefix of every self-check panic (what a binary's panic guard looks for).
pub const MARKER: &str = "EMISSION SELF-CHECK FAILED";

/// The exit status a binary uses when a caught panic was a self-check failure.
pub const EXIT_STATUS: i32 = 86;

/// Whether a caught panic message is a self-check failure.
pub fn is_failure(msg: &str) -> bool {
    msg.contains(MARKER)
}

/// For a binary that catches panics per request: a self-check failure is NOT recoverable — print
/// it and exit, so no caller can mistake it for one bad request and carry on.
pub fn exit_if_failure(msg: &str) {
    if is_failure(msg) {
        eprintln!("{msg}");
        std::process::exit(EXIT_STATUS);
    }
}

#[cold]
#[inline(never)]
fn fail(what: &str, viewer: &str, detail: String) -> ! {
    panic!("{MARKER} [{what}] viewer={viewer}\n{detail}");
}

fn viewer_name(v: usize) -> String {
    format!("p{}", v + 1)
}

fn opt(s: Option<&str>) -> String {
    match s {
        Some(t) => format!("{t:?}"),
        None => "<not sent>".to_string(),
    }
}

/// Check 1 — an omniscient line: the committed `text` is `render(line)` and parses back to
/// exactly `line`.
pub fn check_omniscient(line: &Line, text: &str) {
    bump(0);
    let rendered = line.render();
    let back = Line::parse(text);
    if rendered == text && back.as_ref() == Ok(line) {
        return;
    }
    fail(
        "round-trip",
        "omniscient",
        format!(
            "  line (typed):      {line:?}\n  render(line):      {rendered:?}\n  committed text:    {text:?}\n  \
             parse(text):       {back:?}\n  render(parse(..)): {:?}",
            back.as_ref().map(Line::render)
        ),
    );
}

/// Which side ONLY receives this omniscient line, if any — the owner-only rules (gen-3
/// Pressure's `addSplit` `-ability …|[silent]`; the Intimidate-vs-Substitute `-hint`, whose
/// owner is the caller's fact because the line itself names no mon). `Err` = an owner-only line
/// whose owner nobody knows (a leak waiting to happen).
fn owner_only(typed: &Line, hint_owner: Option<usize>) -> Result<Option<usize>, &'static str> {
    if typed.kw == Kw::Ability && matches!(typed.fields.last(), Some(Field::Tag(t)) if t == "[silent]") {
        return typed.owner_side().map(|s| Some(s as usize)).ok_or("a [silent] -ability line names no side");
    }
    if typed.kw == Kw::Hint && typed.render().contains("Intimidate does not activate") {
        return hint_owner.map(Some).ok_or("an owner-only Intimidate -hint with no known owner");
    }
    Ok(None)
}

/// The HP field index of an HP-bearing keyword (`switch`/`drag` 2; `-damage`/`-heal`/`-sethp` 1).
fn hp_idx(kw: Kw) -> Option<usize> {
    match kw {
        Kw::Switch | Kw::Drag => Some(2),
        Kw::Damage | Kw::Heal | Kw::Sethp => Some(1),
        _ => None,
    }
}

/// The spec-level SECRET invariant for one viewer's parsed render of `typed`: in a percent
/// format, an HP field the viewer does not own is `pct/100` (never the owner's exact `hp/max`).
fn secret_hp_leak(typed: &Line, got: &Line, viewer: usize, report_percent: bool) -> Option<String> {
    if !report_percent {
        return None;
    }
    let i = hp_idx(typed.kw)?;
    let owner = typed.owner_side()? as usize;
    if owner == viewer {
        return None;
    }
    let Some(Field::Hp(Hp::Alive { hp, max, .. })) = typed.fields.get(i) else { return None };
    if *max == 100 {
        return None; // the exact and the percent forms coincide — nothing to hide
    }
    match got.fields.get(i) {
        // A percent — right or wrong, it is not the secret (a wrong one fails the per-viewer check).
        Some(Field::Hp(Hp::Alive { max: 100, .. })) | Some(Field::Hp(Hp::Fainted)) => None,
        other => Some(format!("the owner's exact HP {hp}/{max} (or a non-percent form) reached the non-owner as {other:?}")),
    }
}

/// Check 2 — ONE viewer's render of the omniscient line `omni`. `sent` is what the production
/// fold gave `viewer` (`None` = withheld); `other` is what it gave the other viewer (for the
/// failure message only). `hint_owner` is the caller's owner of an Intimidate `-hint`.
pub fn check_side(
    omni: &str,
    viewer: usize,
    hint_owner: Option<usize>,
    report_percent: bool,
    sent: Option<&str>,
    other: Option<&str>,
) {
    bump(1);
    let v = viewer_name(viewer);
    let renders = |expected: Option<&str>| {
        format!(
            "  omniscient line:   {omni:?}\n  {v} render:         {}\n  p{} render:         {}\n  {v} expected:       {}",
            opt(sent),
            2 - viewer,
            opt(other),
            opt(expected),
        )
    };
    let typed = match Line::parse(omni) {
        Ok(l) => l,
        Err(e) => fail("unparseable", &v, format!("{}\n  parse error: {e}", renders(None))),
    };
    let only = match owner_only(&typed, hint_owner) {
        Ok(o) => o,
        Err(why) => fail("owner-unknown", &v, format!("{}\n  {why}", renders(None))),
    };
    let should_see = only.map_or(true, |o| o == viewer);
    let expected = side_view(&typed, viewer as u8, report_percent);
    let expected_text = expected.render();
    match (should_see, sent) {
        (false, None) => {}
        (false, Some(_)) => fail(
            "SECRET LEAK",
            &v,
            format!("{}\n  an owner-only line (owner p{}) reached the other viewer", renders(None), only.unwrap() + 1),
        ),
        (true, None) => fail("withheld", &v, format!("{}\n  the viewer is owed this line", renders(Some(&expected_text)))),
        (true, Some(text)) => {
            let got = match Line::parse(text) {
                Ok(l) => l,
                Err(e) => fail("unparseable", &v, format!("{}\n  parse error: {e}", renders(Some(&expected_text)))),
            };
            if let Some(leak) = secret_hp_leak(&typed, &got, viewer, report_percent) {
                fail("SECRET LEAK", &v, format!("{}\n  {leak}", renders(Some(&expected_text))));
            }
            if got != expected || text != expected_text {
                fail(
                    "per-viewer",
                    &v,
                    format!("{}\n  parse(render):     {got:?}\n  expected (typed):  {expected:?}", renders(Some(&expected_text))),
                );
            }
        }
    }
}

/// Check 3a — one `|split|pN` triple of the replay family's `battle.log` shape: `secret` is the
/// owner's line, `shared` the other viewer's (EMPTY when the line is owner-only).
pub fn check_split(omni: &str, owner: usize, hint_owner: Option<usize>, report_percent: bool, secret: &str, shared: &str) {
    bump(2);
    check_side(omni, owner, hint_owner, report_percent, Some(secret), Some(shared));
    let other = 1 - owner;
    let only = Line::parse(omni).ok().and_then(|t| owner_only(&t, hint_owner).ok().flatten());
    let other_sent = if only == Some(owner) {
        if !shared.is_empty() {
            fail(
                "SECRET LEAK",
                &viewer_name(other),
                format!("  omniscient line:   {omni:?}\n  |split|p{} secret: {secret:?}\n  shared:            {shared:?}\n  \
                         an owner-only line's shared half must be empty", owner + 1),
            );
        }
        None
    } else {
        Some(shared)
    };
    check_side(omni, other, hint_owner, report_percent, other_sent, Some(secret));
}

/// Check 3b — a line the `battle.log` shape broadcasts UN-split: both viewers must be owed it and
/// read it identically (a line that differs per viewer, or is owner-only, must be split).
pub fn check_broadcast(omni: &str, report_percent: bool) {
    bump(2);
    for viewer in 0..2 {
        check_side(omni, viewer, None, report_percent, Some(omni), Some(omni));
    }
}

/// The FAULT-INJECTION hook the binaries' exit guard is pinned with
/// (`tests/emission_check_test.rs`): with `POKESIM_EMISSION_SELFCHECK_INJECT=1` in a self-check
/// build, the first bridge frame fails the check. Read once; compiled out with every other call.
fn injected() -> bool {
    static INJECT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INJECT.get_or_init(|| std::env::var("POKESIM_EMISSION_SELFCHECK_INJECT").as_deref() == Ok("1"))
}

/// Check 4 — a frame the BRIDGE builds for one side (no omniscient source): it parses under a
/// known keyword and re-renders to the same bytes; a `|request|` carries that side's id and a
/// forced-Struggle `|-activate|` names that side's mon (both are the side's secrets).
pub fn check_frame(side: usize, text: &str) {
    bump(3);
    let v = viewer_name(side);
    if injected() {
        fail("injected", &v, format!("  frame: {text:?}\n  POKESIM_EMISSION_SELFCHECK_INJECT=1"));
    }
    let line = match Line::parse(text) {
        Ok(l) => l,
        Err(e) => fail("unparseable-frame", &v, format!("  frame: {text:?}\n  parse error: {e}")),
    };
    let back = line.render();
    if back != text {
        fail("frame-round-trip", &v, format!("  frame:          {text:?}\n  render(parse):  {back:?}"));
    }
    let owner_mismatch = match line.kw {
        Kw::Request => {
            let want = format!("\"id\":\"p{}\"", side + 1);
            let side_obj = text.find("\"side\":{").map(|i| &text[i..]);
            match side_obj {
                Some(s) => !s.contains(&want),
                None => true,
            }
        }
        Kw::Activate => line.owner_side().map_or(false, |o| o as usize != side),
        _ => false,
    };
    if owner_mismatch {
        fail("SECRET LEAK", &v, format!("  frame: {text:?}\n  a one-side frame shipped to a side it does not belong to"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(f);
        std::panic::set_hook(prev);
        let e = r.expect_err("the self-check must panic");
        e.downcast_ref::<String>().cloned().unwrap_or_default()
    }

    #[test]
    fn a_canonical_line_passes_and_a_non_canonical_one_panics_with_both_renders() {
        let l = Line::parse("|-damage|p2a: Snorlax|116/524 slp|[from] Leech Seed|[of] p1a: Venusaur").unwrap();
        check_omniscient(&l, &l.render());
        // A `|` inside one field is two fields once re-read — not canonical.
        let bad = Line::new(Kw::Start, vec![Field::Ident(Ident::active(0, "Gengar")), Field::text("Substitute|x")]);
        let msg = panics(move || check_omniscient(&bad, &bad.render()));
        assert!(msg.starts_with(MARKER), "{msg}");
        assert!(msg.contains("viewer=omniscient") && msg.contains("render(line)") && msg.contains("parse(text)"), "{msg}");
    }

    use crate::core_events::line::Ident;

    #[test]
    fn a_percent_fold_passes_for_both_viewers() {
        let omni = "|-damage|p1a: Blissey|300/651|[from] psn";
        check_side(omni, 0, None, true, Some(omni), Some("|-damage|p1a: Blissey|47/100|[from] psn"));
        check_side(omni, 1, None, true, Some("|-damage|p1a: Blissey|47/100|[from] psn"), Some(omni));
        // A debug format shows exact HP to both.
        check_side(omni, 1, None, false, Some(omni), Some(omni));
    }

    #[test]
    fn the_owners_exact_hp_at_the_other_viewer_is_a_secret_leak() {
        let omni = "|-damage|p1a: Blissey|300/651";
        let msg = panics(move || check_side(omni, 1, None, true, Some(omni), Some(omni)));
        assert!(msg.contains("[SECRET LEAK] viewer=p2") && msg.contains("300/651"), "{msg}");
    }

    #[test]
    fn an_owner_only_line_at_the_other_viewer_is_a_secret_leak_and_withholding_it_from_the_owner_fails() {
        let omni = "|-ability|p2a: Skarmory|Pressure|[silent]";
        check_side(omni, 1, None, true, Some(omni), None);
        check_side(omni, 0, None, true, None, Some(omni));
        let msg = panics(move || check_side(omni, 0, None, true, Some(omni), Some(omni)));
        assert!(msg.contains("[SECRET LEAK] viewer=p1"), "{msg}");
        let msg = panics(move || check_side(omni, 1, None, true, None, None));
        assert!(msg.contains("[withheld] viewer=p2"), "{msg}");
    }

    #[test]
    fn an_intimidate_hint_needs_a_known_owner() {
        let omni = "|-hint|In Gen 3, Intimidate does not activate if every target has a Substitute.";
        check_side(omni, 0, Some(0), true, Some(omni), None);
        let msg = panics(move || check_side(omni, 0, None, true, Some(omni), Some(omni)));
        assert!(msg.contains("[owner-unknown]"), "{msg}");
    }

    #[test]
    fn a_wrong_percent_is_a_per_viewer_failure_not_a_leak() {
        let omni = "|switch|p2a: Zapdos|Zapdos|300/383";
        let msg = panics(move || check_side(omni, 0, None, true, Some("|switch|p2a: Zapdos|Zapdos|78/100"), Some(omni)));
        // ceil(30000/383) = 79
        assert!(msg.contains("[per-viewer] viewer=p1"), "{msg}");
        assert!(msg.contains("p1 expected:") && msg.contains("79/100"), "{msg}");
    }

    #[test]
    fn a_split_triple_is_checked_both_ways() {
        let omni = "|-heal|p2a: Metagross|133/328|[from] item: Leftovers";
        check_split(omni, 1, None, true, omni, "|-heal|p2a: Metagross|41/100|[from] item: Leftovers");
        let msg = panics(move || check_split(omni, 1, None, true, omni, omni));
        assert!(msg.contains("[SECRET LEAK] viewer=p1"), "{msg}");
        let p = "|-ability|p1a: Dusclops|Pressure|[silent]";
        check_split(p, 0, None, true, p, "");
        let msg = panics(move || check_split(p, 0, None, true, p, p));
        assert!(msg.contains("[SECRET LEAK] viewer=p2"), "{msg}");
    }

    #[test]
    fn an_un_split_line_must_read_the_same_to_both_viewers() {
        check_broadcast("|move|p1a: Gengar|Thunderbolt|p2a: Skarmory", true);
        let msg = panics(|| check_broadcast("|-damage|p2a: Skarmory|100/334", true));
        assert!(msg.contains("[SECRET LEAK] viewer=p1"), "{msg}");
    }

    #[test]
    fn a_frame_for_the_wrong_side_is_a_secret_leak() {
        check_frame(0, "|request|{\"active\":[],\"side\":{\"name\":\"a\",\"id\":\"p1\",\"pokemon\":[]}}");
        let msg = panics(|| check_frame(1, "|request|{\"active\":[],\"side\":{\"name\":\"a\",\"id\":\"p1\",\"pokemon\":[]}}"));
        assert!(msg.contains("[SECRET LEAK] viewer=p2"), "{msg}");
        check_frame(1, "|-activate|p2a: Snorlax|move: Struggle");
        let msg = panics(|| check_frame(0, "|-activate|p2a: Snorlax|move: Struggle"));
        assert!(msg.contains("[SECRET LEAK] viewer=p1"), "{msg}");
        check_frame(0, "|error|[Invalid choice] Can't switch: The active Pokémon is trapped");
        let msg = panics(|| check_frame(0, "|-dynamax|p1a: X"));
        assert!(msg.contains("[unparseable-frame]"), "{msg}");
    }

    #[test]
    fn the_marker_is_what_a_binary_guard_recognizes() {
        assert!(is_failure(&format!("{MARKER} [x] viewer=p1")));
        assert!(!is_failure("some other panic"));
        assert_eq!(ENABLED, cfg!(any(debug_assertions, feature = "emission-selfcheck")));
    }

    #[test]
    fn every_check_counts_itself() {
        let before = counts();
        let omni = "|-damage|p1a: Blissey|300/651";
        check_omniscient(&Line::parse(omni).unwrap(), omni);
        check_side(omni, 0, None, true, Some(omni), None);
        check_split(omni, 0, None, true, omni, "|-damage|p1a: Blissey|47/100");
        check_frame(0, "|error|x");
        let after = counts();
        // Tests run in parallel, so only a lower bound is exact.
        assert!(after[0] > before[0] && after[1] >= before[1] + 3 && after[2] > before[2] && after[3] > before[3]);
        assert!(summary().starts_with("emission_selfcheck omniscient="));
    }
}
