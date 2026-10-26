//! The Custom Game EV-BALANCE warning (`gen3_ev_balance_bigerror_v1`): `Battle.start()` runs
//! `checkEVBalance()` in a `debug` format, and when exactly one side holds a mon whose EVs sum past
//! 510 it adds `|bigerror|Warning: One player isn't adhering to a 510 EV limit, and the other player
//! is.` after the rules (`sim/battle.ts` checkEVBalance). Found by the B9 Attract golden, whose
//! Protect boards field a 677-EV set; the golden rows pin the bytes against the sim, this pins the
//! three branches.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

const WARN: &str = "|bigerror|Warning: One player isn't adhering to a 510 EV limit, and the other player is.";

fn framing(format: &str, p1_evs: &str, p2_evs: &str) -> Vec<String> {
    let d = Dex::for_gen(3);
    let set = |evs: &str| format!("Snorlax|Snorlax|Leftovers|ThickFat|splash|Hardy|{evs}|M||||");
    let opts = BattleOptions {
        format_id: format.into(),
        seed: Some("1,2,3,4".into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(set(p1_evs)) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(set(p2_evs)) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let (_o, lines) = b.state_mut().unwrap().run_full_battle_logged(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    pokesim::bridge::reframe(&lines, format)
}

#[test]
fn ev1_one_side_over_510_warns_in_custom_game_only() {
    let over = "252,252,252,0,0,0";
    let under = "85,85,85,85,85,85";
    let f = framing("gen3customgame", over, under);
    let rule = f.iter().position(|l| l.starts_with("|rule|HP Percentage Mod")).unwrap();
    assert_eq!(f[rule + 1], WARN, "the warning follows the rules: {f:#?}");
    assert!(!framing("gen3customgame", over, over).iter().any(|l| l == WARN), "both over: no warning");
    assert!(!framing("gen3customgame", under, under).iter().any(|l| l == WARN), "both under: no warning");
    assert!(!framing("gen3ubers", over, under).iter().any(|l| l == WARN), "not a debug format: no warning");
}
