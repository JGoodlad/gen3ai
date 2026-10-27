//! The FORMAT TABLE against Showdown (`gen3_format_rules_v1`, gen3 full parity B1).
//!
//! `src/format.rs` is hand-written; `tests/vectors/gen3_formats.json` is what
//! `harness/dump_gen3_formats.js` reads off Showdown's resolved rule tables and a started battle
//! per format (`--check` is its drift gate). This test holds the two equal, and pins the
//! refusals: a format, level, team size or move count the engine does not model is an `Err` at
//! construction, never a battle run as something else.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::format::{rules, FORMATS, REFUSED};
use pokesim::json::Json;

fn vector() -> Json {
    Json::parse(include_str!("vectors/gen3_formats.json")).expect("gen3_formats.json is JSON")
}

fn strs(v: &Json) -> Vec<String> {
    v.as_array().expect("array").iter().map(|x| x.as_str().expect("string").to_string()).collect()
}

fn flag(row: &Json, key: &str) -> bool {
    row.get(key).and_then(|x| x.as_bool()).unwrap_or_else(|| panic!("row lacks bool {key}"))
}

/// FR1 — every format Showdown defines for the gen-3 singles engine is in the table with
/// Showdown's rules, or refused with a reason; and the table holds nothing else.
#[test]
fn the_format_table_is_showdowns() {
    let v = vector();
    let supported = v.get("supported").and_then(|x| x.as_array()).expect("supported");
    let ids: Vec<&str> = supported.iter().map(|r| r.get("id").and_then(|x| x.as_str()).expect("id")).collect();
    let table: Vec<&str> = FORMATS.iter().map(|r| r.id).collect();
    assert_eq!(table, ids, "the Rust table's formats (in order) differ from Showdown's");
    for row in supported {
        let id = row.get("id").and_then(|x| x.as_str()).unwrap();
        let r = rules(id).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(r.tier, row.get("tier").and_then(|x| x.as_str()).unwrap(), "{id}: |tier|");
        let want_rules = strs(row.get("rules").unwrap());
        let got_rules: Vec<String> = r.rules.iter().map(|s| s.to_string()).collect();
        assert_eq!(got_rules, want_rules, "{id}: the |rule| lines");
        assert_eq!(r.sleep_clause, flag(row, "sleep_clause"), "{id}: sleep_clause");
        assert_eq!(r.freeze_clause, flag(row, "freeze_clause"), "{id}: freeze_clause");
        assert_eq!(r.species_clause, flag(row, "species_clause"), "{id}: species_clause");
        assert_eq!(r.beatup_nicknames, flag(row, "beatup_nicknames"), "{id}: beatup_nicknames");
        assert_eq!(r.ability_leak, flag(row, "ability_leak"), "{id}: ability_leak");
        assert_eq!(r.exact_hp, flag(row, "exact_hp"), "{id}: exact_hp");
        assert_eq!(r.endless_battle_clause, flag(row, "endless_battle_clause"), "{id}: endless_battle_clause");
        let banned: Vec<String> = r.leak_banned_abilities.iter().map(|s| s.to_string()).collect();
        assert_eq!(banned, strs(row.get("leak_banned_abilities").unwrap()), "{id}: leak_banned_abilities");
        // The SetStatus shuffle is a tie of exactly two clause handlers.
        let handlers = strs(row.get("set_status_clause_handlers").unwrap());
        assert_eq!(r.set_status_shuffle(), handlers.len() == 2, "{id}: the SetStatus clause tie ({handlers:?})");
        assert_eq!(
            r.pseudo_weather().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            handlers,
            "{id}: the clause pseudo-weathers"
        );
        // The engine refuses what a format allows past these, so a format that allowed less
        // would need its own refusal.
        assert!(row.get("max_team_size").and_then(|x| x.as_f64()).unwrap() >= 6.0, "{id}: max team size");
        assert!(row.get("max_move_count").and_then(|x| x.as_f64()).unwrap() >= 4.0, "{id}: max move count");
    }
    let refused = v.get("refused").and_then(|x| x.as_array()).expect("refused");
    let want: Vec<&str> = refused.iter().map(|r| r.get("id").and_then(|x| x.as_str()).unwrap()).collect();
    let got: Vec<&str> = REFUSED.iter().map(|(id, _)| *id).collect();
    assert_eq!(got, want, "the refused formats");
}

fn opts(format: &str, p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: format.to_string(),
        seed: Some("1,2,3,4".to_string()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.to_string()) },
    }
}

const SNORLAX: &str = "|Snorlax|Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";

fn start_err(o: &BattleOptions) -> String {
    let d = Dex::for_gen(3);
    match Battle::start_with_switchins(o, &d) {
        Ok(_) => panic!("{} started; it must be refused", o.format_id),
        Err(e) => e,
    }
}

/// FR2 — a format the table does not hold is REFUSED, never run as gen3ou (the old
/// `!ends_with("customgame")` rule ran every one of these as OU). A display-name spelling of a
/// supported format still resolves (`toID`).
#[test]
fn an_unmodeled_format_is_refused() {
    for f in ["gen31v1", "gen3doublesou", "gen3doublescustomgame", "gen3adv200", "gen3frlgou", "gen4ou", "gen3ouu", ""] {
        let e = start_err(&opts(f, SNORLAX, SNORLAX));
        assert!(e.contains("is not supported"), "{f}: {e}");
    }
    let d = Dex::for_gen(3);
    let b = Battle::start_with_switchins(&opts("[Gen 3] Ubers", SNORLAX, SNORLAX), &d).expect("[Gen 3] Ubers is gen3ubers");
    assert_eq!(b.state().unwrap().rules.id, "gen3ubers");
}

/// FR3 — a level the engine does not model is refused, and a level-0 field reads as 100 the way
/// Showdown's ctor reads it (`set.level || 100`). Before: `parse::<u8>("300")` failed and the
/// `unwrap_or(100)` silently built a level-100 mon.
#[test]
fn levels_outside_1_to_100_are_refused_and_0_is_100() {
    let lv = |n: &str| format!("|Snorlax|Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M|||{n}|");
    for n in ["300", "101", "9999"] {
        let e = start_err(&opts("gen3customgame", &lv(n), SNORLAX));
        assert!(e.contains(&format!("level {n} is not supported")), "level {n}: {e}");
    }
    let d = Dex::for_gen(3);
    let zero = Battle::start_with_switchins(&opts("gen3customgame", &lv("0"), SNORLAX), &d).expect("level 0");
    let hundred = Battle::start_with_switchins(&opts("gen3customgame", &lv(""), SNORLAX), &d).expect("level 100");
    let (a, b) = (&zero.state().unwrap().sides[0].pokemon[0], &hundred.state().unwrap().sides[0].pokemon[0]);
    assert_eq!((a.level, a.maxhp), (100, b.maxhp), "a level-0 field is a level-100 mon");
}

/// FR4 — Custom Game allows 24 mons: no gate has run more than 6, so a 7th is refused. Its 24-move
/// limit is the engine's (a 6-move set is pinned by `regression_test.rs` WB1); a 25th is refused.
#[test]
fn oversize_teams_and_movesets_are_refused() {
    let seven = vec![
        "|Snorlax|||splash|Hardy||M||||",
        "|Blissey|||splash|Hardy||F||||",
        "|Chansey|||splash|Hardy||F||||",
        "|Skarmory|||splash|Hardy||M||||",
        "|Suicune|||splash|Hardy||N||||",
        "|Raikou|||splash|Hardy||N||||",
        "|Entei|||splash|Hardy||N||||",
    ]
    .join("]");
    let e = start_err(&opts("gen3customgame", &seven, SNORLAX));
    assert!(e.contains("7 Pokémon is not supported"), "{e}");
    let moves: Vec<&str> = ["splash", "tackle", "rest", "curse", "protect"].iter().copied().cycle().take(25).collect();
    let many = format!("|Snorlax|Leftovers|Immunity|{}|Hardy|85,85,85,85,85,85|M||||", moves.join(","));
    let e = start_err(&opts("gen3customgame", &many, SNORLAX));
    assert!(e.contains("25 moves is not supported"), "{e}");
}

/// FR5 — the committed vector is what Showdown says NOW (`dump_gen3_formats.js --check`): a
/// Showdown update that changes a format's rules or adds a gen-3 singles format fails here until
/// the table and the vector are re-derived. Like the handler audit, it fails loudly when node or
/// the dist is unavailable, never skips.
#[test]
fn the_format_vector_is_current() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = std::process::Command::new("node")
        .arg("harness/dump_gen3_formats.js")
        .arg("--check")
        .current_dir(dir)
        .output()
        .expect("node must be runnable for the format drift gate (see root CLAUDE.md worktree setup)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success() && stderr.contains("gen3 format table OK"), "format drift gate:\n{stderr}");
}
