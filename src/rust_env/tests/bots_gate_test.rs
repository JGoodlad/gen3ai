//! THE BOT GATE on a banked corpus (M5 Lane F). The corpus is written by
//! `src/utils/rust_env/bot_corpus.py`; `src/utils/rust_env/bots_gate_test.py` decompresses the
//! committed COMMIT tier (routine) or records a MILESTONE corpus (`slow`) and runs this with
//! `POKESIM_BOTS_CORPUS=<json>` (`--ignored`: without a corpus there is nothing to hold equal, and a
//! gate that silently passes on nothing is the failure we refuse). The report is the line after
//! `BOTS_GATE `, and `POKESIM_BOTS_EXPECT_FAIL=1` inverts the verdict (the teeth runs).

use pokesim_env::bots::gate;

#[test]
#[ignore = "needs POKESIM_BOTS_CORPUS (run by src/utils/rust_env/bots_gate_test.py)"]
fn every_banked_bot_decision_is_equal() {
    let path = std::env::var("POKESIM_BOTS_CORPUS").expect("POKESIM_BOTS_CORPUS");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let rep = gate::run(&text).expect("a well-formed corpus");
    println!("BOTS_GATE {}", rep.json());
    assert!(rep.decisions() > 0, "the corpus held no bot decision — nothing was compared");
    let expect_fail = std::env::var("POKESIM_BOTS_EXPECT_FAIL").is_ok_and(|v| v == "1");
    if expect_fail {
        assert!(rep.mismatches() > 0, "a perturbed corpus PASSED — the gate has no teeth");
    } else {
        assert_eq!(rep.mismatches(), 0, "{:#?}", rep.examples);
    }
}
