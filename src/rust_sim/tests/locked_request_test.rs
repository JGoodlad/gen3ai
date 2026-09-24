//! `gen3_locked_request_move_v1` — a move-LOCKED `|request|` names the LOCKED move.
//!
//! `bridge::serialize_active_with_disabled_source` rendered the single locked entry from the
//! two-turn charge ONLY and fell back to `"solarbeam"` for every other lock `move_locked()`
//! covers — so a Rollout, Ice Ball, Outrage, Thrash, Petal Dance or Uproar lock shipped
//! `{"moves":[{"move":"Solar Beam","id":"solarbeam"}],"trapped":true}` on the TRAINING transport,
//! and poke-env's live player ASSERTS on a request move its mon does not have
//! (`Pokemon.available_moves_from_request`). Found by the Rust Core parity harness's procedural
//! milestone sweep (a Registeel locked into Rollout); zero exposure in the 719-team training pool,
//! but live on the in-repo websocket front end every external-anchor read plays over.
//!
//! THE ORACLE is the pinned Showdown: each expected `active` block below was printed by
//! `BattleStream` + `getPlayerStreams` on the SAME format, seed, teams and choices (p1 uses move 1
//! once; the next p1 request is read).

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;

const FOE: &str = "Snorlax|||immunity|splash,bodyslam,rest,curse|Careful|252,,4,,252,|N||||";

fn locked_active_block(team: &str) -> String {
    let dex = Dex::for_gen(3);
    let opts = bridge_opts("gen3customgame", "1,2,3,4".to_string(), team, FOE);
    let mut sess = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
    for side in 0..2 {
        sess.feed_cmd(Cmd { side, choice: parse_choice("move 1").unwrap() }, &dex);
    }
    let req = sess.active_request_json(0).expect("p1 has a request").to_string();
    let at = req.find("\"active\":").expect("an active block");
    let tail = &req[at + "\"active\":".len()..];
    tail[..tail.find(",\"side\"").expect("side follows active")].to_string()
}

#[test]
fn every_move_lock_requests_the_locked_move_as_showdown_does() {
    let cases = [
        ("Registeel|||clearbody|rollout,explosion,seismictoss,counter|Careful|252,,4,,252,|N||||",
         r#"[{"moves":[{"move":"Rollout","id":"rollout"}],"trapped":true}]"#),
        ("Regice|||clearbody|iceball,explosion,thunderbolt,rest|Careful|252,,4,,252,|N||||",
         r#"[{"moves":[{"move":"Ice Ball","id":"iceball"}],"trapped":true}]"#),
        ("Salamence|||intimidate|outrage,earthquake,fireblast,rockslide|Naive|,252,,4,,252|M||||",
         r#"[{"moves":[{"move":"Outrage","id":"outrage"}],"trapped":true}]"#),
        ("Tauros|||intimidate|thrash,earthquake,rockslide,return|Jolly|,252,,4,,252|M||||",
         r#"[{"moves":[{"move":"Thrash","id":"thrash"}],"trapped":true}]"#),
        ("Vileplume|||chlorophyll|petaldance,sleeppowder,sludgebomb,moonlight|Modest|252,,4,252,,|M||||",
         r#"[{"moves":[{"move":"Petal Dance","id":"petaldance"}],"trapped":true}]"#),
        ("Exploud|||soundproof|uproar,earthquake,fireblast,return|Modest|252,,4,252,,|M||||",
         r#"[{"moves":[{"move":"Uproar","id":"uproar"}],"trapped":true}]"#),
    ];
    for (team, want) in cases {
        let got = locked_active_block(team);
        assert_eq!(got, want, "{}: the locked request must name the LOCKED move",
            team.split('|').next().unwrap());
    }
}

#[test]
fn a_locked_mon_answered_BY_NAME_resolves_and_the_battle_moves_on() {
    // poke-env answers the single-entry request with `move rollout` (the id it was offered). The
    // resolver matched a locked mon's name against the two-turn slot ONLY, so it refused the
    // lock's own move and the bridge failed loud ("unresolvable choice … MoveName(\"rollout\")")
    // — the sequel the request fix exposed on the same procedural battle.
    let dex = Dex::for_gen(3);
    for (team, name) in [
        ("Registeel|||clearbody|rollout,explosion,seismictoss,counter|Careful|252,,4,,252,|N||||", "rollout"),
        ("Salamence|||intimidate|outrage,earthquake,fireblast,rockslide|Naive|,252,,4,,252|M||||", "outrage"),
        ("Exploud|||soundproof|uproar,earthquake,fireblast,return|Modest|252,,4,252,,|M||||", "uproar"),
    ] {
        let opts = bridge_opts("gen3customgame", "1,2,3,4".to_string(), team, FOE);
        let mut sess = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
        for side in 0..2 {
            sess.feed_cmd(Cmd { side, choice: parse_choice("move 1").unwrap() }, &dex);
        }
        let turn = sess.turn();
        sess.feed_cmd(Cmd { side: 0, choice: parse_choice(&format!("move {name}")).unwrap() }, &dex);
        sess.feed_cmd(Cmd { side: 1, choice: parse_choice("move 1").unwrap() }, &dex);
        assert!(sess.fatal().is_none(), "{name}: {:?}", sess.fatal());
        assert!(sess.turn() > turn || sess.is_ended(), "{name}: the locked turn did not resolve");
    }
}
