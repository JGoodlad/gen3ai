//! Unit pins of the encoder's own arithmetic and invariants (the byte parity against the Python
//! encoder is slice O, `agents/battle/rust_core_parity_obs.py`).

use super::layout::*;
use super::*;

#[test]
fn the_generated_layout_tiles_the_row() {
    assert_eq!(OFFSET_OPP_TEAM, OFFSET_OUR_TEAM + TEAM_SIZE * POKEMON_FULL_DIM);
    assert_eq!(OFFSET_CONTEXT, OFFSET_OPP_TEAM + TEAM_SIZE * POKEMON_FULL_DIM);
    assert_eq!(OFFSET_GLOBAL, OFFSET_CONTEXT + 2 * ACTIVE_CONTEXT_DIM);
    assert_eq!(OFFSET_REACTIVE, OFFSET_GLOBAL + GLOBAL_ENV_DIM);
    assert_eq!(OFFSET_PAIR_HISTORY, OFFSET_REACTIVE + REACTIVE_DIM);
    assert_eq!(OFFSET_EVENT_WINDOW, OFFSET_PAIR_HISTORY + PAIR_HISTORY_DIM);
    assert_eq!(OBS_DIM, OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM);
    assert_eq!(VOLATILE_SLOTS.len(), VOLATILES_DIM);
    assert_eq!(POKEMON_ACTIVE_OFFSET + 1, POKEMON_FULL_DIM, "the active flag is LAST in the slot");
    // the lookup tables the encoder binary-searches are sorted
    assert!(VOLATILE_TO_SLOT.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(NOT_A_VOLATILE.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn a_test_build_prefills_the_nan_poison() {
    let mut row = [0.0f32; OBS_DIM];
    prefill(&mut row);
    assert!(NAN_POISON);
    assert!(row.iter().all(|x| x.is_nan()), "an unwritten cell must read NaN in a test build");
}

#[test]
fn the_protect_odds_are_the_floored_doubling() {
    assert_eq!(slot_protect(0), 1.0);
    assert_eq!(slot_protect(1), 0.5);
    assert_eq!(slot_protect(2), 0.25);
    assert_eq!(slot_protect(3), 0.125);
    assert_eq!(slot_protect(9), 0.125);
}

fn slot_protect(k: i64) -> f64 {
    super::slot::protect_for_test(k)
}

#[test]
fn the_cant_reason_id_normalises_like_gen3_effects() {
    assert_eq!(cant_reason_id(None).unwrap(), 0);
    assert_eq!(cant_reason_id(Some("slp")).unwrap(), 1);
    assert_eq!(cant_reason_id(Some("move: Taunt")).unwrap(), 1 + CANT_REASONS_LIVE.iter().position(|r| *r == "taunt").unwrap());
    assert_eq!(cant_reason_id(Some("ability: Truant")).unwrap(), 1 + CANT_REASONS_LIVE.iter().position(|r| *r == "truant").unwrap());
    assert_eq!(cant_reason_id(Some("Focus Punch")).unwrap(), 1 + CANT_REASONS_LIVE.iter().position(|r| *r == "focuspunch").unwrap());
    assert!(matches!(cant_reason_id(Some("bogus")), Err(CoreError::Refusal { exc: PyExc::UnknownCantReasonError, .. })));
}

#[test]
fn an_unclassified_volatile_is_refused_and_a_not_a_volatile_encodes_nothing() {
    let mut out = [f32::NAN; VOLATILES_DIM];
    volatiles(&[("healbell".to_string(), 0)], &mut out).unwrap();
    assert!(out.iter().all(|x| *x == 0.0));
    let err = volatiles(&[("mudsport".to_string(), 0)], &mut out).unwrap_err();
    assert!(matches!(err, CoreError::Refusal { exc: PyExc::UnknownVolatileError, .. }), "{err:?}");
    // a counter keeps its max level: stockpile2 over stockpile1
    volatiles(&[("stockpile1".to_string(), 0), ("stockpile2".to_string(), 0)], &mut out).unwrap();
    let i = VOLATILE_SLOTS.iter().position(|s| *s == "stockpile").unwrap();
    assert_eq!(out[i], (2.0f64 / 3.0) as f32);
}
