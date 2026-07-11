//! Team tests:
//!   - `team_golden_roundtrip` — differential vs Showdown's `Teams.pack`/
//!     `Teams.unpack` (golden from `harness/gen_team_golden.js`): for every
//!     case, Rust `unpack(PACKED)` reproduces every canonical field, AND Rust
//!     `pack(that)` reproduces the expected packed string.
//!   - `team_smoke` — a few authoritative hand-checks independent of the golden
//!     (the EV/IV/level/happiness defaults the bit-for-bit bridge depends on).
//!
//! Mirrors the structure of `tests/dex_test.rs`.

use pokesim::dex::{to_id, Dex};
use pokesim::team::{pack, unpack, PokemonSet};

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// Render a decoded set into the same TAB key=value form the golden emits, so a
/// single string compare covers every field. Ids are `to_id`-normalized to
/// match the harness (which `toID`s them) — producer/case-agnostic.
fn fields(s: &PokemonSet) -> String {
    let evs = format!("{},{},{},{},{},{}", s.evs[0], s.evs[1], s.evs[2], s.evs[3], s.evs[4], s.evs[5]);
    let ivs = format!("{},{},{},{},{},{}", s.ivs[0], s.ivs[1], s.ivs[2], s.ivs[3], s.ivs[4], s.ivs[5]);
    let moves = s.moves.iter().map(|m| to_id(m)).collect::<Vec<_>>().join(",");
    [
        format!("name={}", s.name),
        format!("species={}", to_id(&s.species)),
        format!("item={}", to_id(&s.item)),
        format!("ability={}", to_id(&s.ability)),
        format!("moves={moves}"),
        format!("nature={}", s.nature),
        format!("evs={evs}"),
        format!("ivs={ivs}"),
        format!("gender={}", s.gender.map_or(String::new(), |c| c.to_string())),
        format!("shiny={}", if s.shiny { "1" } else { "0" }),
        format!("level={}", s.level),
        format!("happiness={}", s.happiness),
        format!("hptype={}", to_id(&s.hp_type)),
    ]
    .join("\t")
}

/// Normalize the golden's `hptype=Grass` to the `to_id`-comparable form so the
/// expected and got strings line up (the golden writes the display hpType).
fn normalize_expected_hptype(line: &str) -> String {
    // Find the `hptype=` field and to_id its value (mirrors `fields`).
    line.split('\t')
        .map(|kv| {
            if let Some(v) = kv.strip_prefix("hptype=") {
                format!("hptype={}", to_id(v))
            } else {
                kv.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\t")
}

#[test]
fn team_golden_roundtrip() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/team_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing team golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_team_golden.js")
    });
    let d = dex();

    // The golden is a stream of (IN, UNPACK, PACK) triples. PACK is the canonical
    // re-pack, which may differ from IN (e.g. a poke-env lowercase input).
    let mut input: Option<String> = None;
    let mut expect_fields: Option<String> = None;
    let mut checked = 0usize;

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tag, rest) = line.split_once('\t').unwrap_or_else(|| panic!("no tab (line {ln})"));
        match tag {
            "IN" => {
                input = Some(rest.to_string());
            }
            "UNPACK" => {
                expect_fields = Some(normalize_expected_hptype(rest));
            }
            "PACK" => {
                let p = input.take().unwrap_or_else(|| panic!("PACK before IN (line {ln})"));
                let exp = expect_fields.take().unwrap_or_else(|| panic!("PACK before UNPACK (line {ln})"));

                // 1) unpack(IN) reproduces every canonical field.
                let sets = unpack(&p, &d).unwrap_or_else(|e| panic!("unpack failed (line {ln}): {e}"));
                assert_eq!(sets.len(), 1, "single-set golden case (line {ln})");
                let got = fields(&sets[0]);
                assert_eq!(got, exp, "unpack fields mismatch (line {ln})\n  input: {p}");

                // 2) pack(unpack(IN)) reproduces the canonical packed string.
                let repacked = pack(&sets, &d);
                assert_eq!(repacked, rest, "repack mismatch (line {ln})\n  input: {p}");

                checked += 1;
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }
    assert!(checked >= 20, "expected the full team golden corpus, got {checked}");
    eprintln!("team golden: {checked} cases (×2 unpack+pack assertions) passed");
}

#[test]
fn team_smoke() {
    let d = dex();

    // A real gen-3 packed mon (the bridge form): Suicune with an explicit
    // non-default IV line (Atk 2 / SpA 30), which MUST survive verbatim.
    let packed = "Suicune||Leftovers|Pressure|CalmMind,HydroPump,IceBeam,HiddenPowerGrass|Timid|56,,,220,,232||,2,,30,,|||";
    let sets = unpack(packed, &d).expect("unpack suicune");
    assert_eq!(sets.len(), 1);
    let s = &sets[0];
    assert_eq!(to_id(&s.species), "suicune");
    assert_eq!(to_id(&s.item), "leftovers");
    assert_eq!(s.evs, [56, 0, 0, 220, 0, 232]); // empty EV slot → 0
    assert_eq!(s.ivs, [31, 2, 31, 30, 31, 31]); // empty IV slot → 31, explicit 2/30 kept
    assert_eq!(s.level, 100, "empty level → default 100");
    assert_eq!(s.happiness, 255, "no happiness → default 255");
    assert!(!s.shiny);
    assert_eq!(s.gender, None);
    // Bit-faithful re-pack.
    assert_eq!(pack(&sets, &d), packed);

    // All-default EV/IV set: both fields collapse to empty on pack.
    let lax = PokemonSet {
        name: "Skarmory".into(),
        species: "Skarmory".into(),
        ability: "Keen Eye".into(),
        moves: vec!["Spikes".into(), "Whirlwind".into()],
        nature: "Impish".into(),
        ..Default::default()
    };
    let p = pack(std::slice::from_ref(&lax), &d);
    assert_eq!(p, "Skarmory|||KeenEye|Spikes,Whirlwind|Impish||||||");
    // And it round-trips back.
    assert_eq!(unpack(&p, &d).unwrap()[0].ivs, [31; 6]);
    assert_eq!(unpack(&p, &d).unwrap()[0].evs, [0; 6]);

    // poke-env producer quirks (our REAL bridge producer): an EMPTY name field
    // with the species in the species field, lowercase ids, and an EXPLICIT "100"
    // level (Showdown omits it). unpack must ingest it and re-pack to
    // Showdown-canonical bytes (case-preserving, level dropped).
    let pe = "|skarmory||keeneye|spikes,whirlwind|Impish|||||100|";
    let pes = unpack(pe, &d).unwrap();
    assert_eq!(pes[0].level, 100, "explicit 100 level accepted");
    assert_eq!(to_id(&pes[0].ability), "keeneye");
    assert_eq!(
        pack(&pes, &d),
        "Skarmory|||KeenEye|Spikes,Whirlwind|Impish||||||",
        "poke-env form re-packs to Showdown-canonical (level omitted, ids cased)"
    );

    // Short poke-env misc tail ",grass," (3 fields) decodes hpType like Showdown's
    // padded ",Grass,,,,".
    let g = unpack("Gengar||Leftovers|Levitate|Explosion|Timid||||||,grass,", &d).unwrap();
    assert_eq!(to_id(&g[0].hp_type), "grass");

    // ']' inside a nickname stays ONE set (sequential parse, not split(']')).
    let bracket = unpack("a]b|Snorlax|Leftovers|Immunity|BodySlam|Adamant||||||", &d).unwrap();
    assert_eq!(bracket.len(), 1, "']' in a nickname does not split the set");
    assert_eq!(bracket[0].name, "a]b");

    // Empty string → empty team; two-set round-trip via ']' separator.
    assert_eq!(unpack("", &d).unwrap().len(), 0);
    let two = pack(&[lax.clone(), lax.clone()], &d);
    assert_eq!(two.matches(']').count(), 1, "two sets joined by one ']'");
    assert_eq!(unpack(&two, &d).unwrap().len(), 2);
}
