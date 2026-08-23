//! `gen3_fire_thaw_ko_keeps_status_v1` — a fire move that KOs a FROZEN target must not thaw it.
//!
//! Showdown's fire-move thaw is `Pokemon.cureStatus()`, and that method opens with
//! `if (!this.hp || !this.status) return false` — so on a KO'ing hit it does NEITHER half: no
//! `|-curestatus|<target>|frz|[msg]` AND no status clear. The corpse keeps `frz` until
//! `Battle.checkFainted` overwrites it with `fnt`; and when the KO ENDS the battle,
//! `checkFainted` never runs at all (`runAction` does `faintMessages(); if (this.ended) return`
//! ahead of it), so `frz` is the last word.
//!
//! The port had the EMIT half right and the CLEAR half wrong, on the reasoning — written into
//! the code — that "a faint overrides it anyway". It does not. `search::outcome_of`'s
//! `outcome.pN.active_status` renders `status_token(mon)` verbatim once the battle has ended,
//! so a deciding faint read `""` where node read `"frz"`. That was the ONE divergence
//! `harness/replay_impl_parity.py` reported on a freshly generated golden (a Jirachi frozen by
//! Ice Beam, then KO'd by Tyranitar's Flamethrower on the last turn of the battle).
//!
//! Both directions are pinned here: the KO case must KEEP the status, and the SURVIVING case
//! must still clear it — a fix that simply stopped thawing would pass the first alone.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn opts(p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("[1,2,3,4]".to_string()),
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    }
}

/// name | species | item | ability | moves | nature | evs | gender | ivs | shiny | level | happiness
const ATTACKER: &str = "Charizard|||blaze|flamethrower|Serious||||||";
/// Snorlax: Normal, so Flamethrower is NEUTRAL (no resist/immunity to muddy the KO), and slower
/// than Charizard, so the fire hit always lands first.
const TARGET: &str = "Snorlax|||immunity|tackle|Serious||||||";

/// Freeze p2's active and set its HP, then run ONE turn of Flamethrower into Tackle.
/// Returns the target's post-turn `(status, fainted)`.
fn thaw_turn(hp: u16) -> (Option<Status>, bool) {
    let d = dex();
    let mut battle = Battle::start_with_switchins(&opts(ATTACKER, TARGET), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let slot = st.sides[1].active;
    st.sides[1].pokemon[slot].status = Some(Status::Freeze);
    st.sides[1].pokemon[slot].hp = hp;
    assert!(
        st.sides[0].pokemon[st.sides[0].active].hp > 0,
        "fixture: the attacker must be alive to swing"
    );

    st.run_turn(0, 0, &d);

    let mon = &st.sides[1].pokemon[slot];
    (mon.status, mon.fainted)
}

/// THE FIX. `cureStatus()` bails on a 0-HP mon, so the corpse still reads `frz` — which is what
/// the referee readout reports when this faint is the one that ends the battle.
#[test]
fn a_fire_move_that_kos_a_frozen_target_leaves_the_freeze_on_the_corpse() {
    let (status, fainted) = thaw_turn(1);
    // NON-VACUITY: the hit must really have KO'd, or "the status survived" is trivially true.
    assert!(fainted, "fixture: Flamethrower must KO the 1-HP target for this pin to mean anything");
    assert_eq!(
        status,
        Some(Status::Freeze),
        "a KO-ing fire move runs NEITHER half of cureStatus() — the corpse keeps `frz` until \
         checkFainted writes `fnt`, and on a battle-ending faint checkFainted never runs. \
         Clearing it here is what made outcome.pN.active_status read \"\" against node's \"frz\"."
    );
}

/// THE CONTROL. A fire move the target SURVIVES still thaws it — the ordinary gen-3 rule, and
/// the half that must not regress while fixing the KO case.
#[test]
fn a_fire_move_the_frozen_target_survives_still_thaws_it() {
    let (status, fainted) = thaw_turn(u16::MAX);
    assert!(!fainted, "fixture: the target must SURVIVE for this to test the thaw");
    assert_eq!(
        status, None,
        "a landed fire move thaws a surviving frozen target (the ordinary gen-3 rule)"
    );
}
