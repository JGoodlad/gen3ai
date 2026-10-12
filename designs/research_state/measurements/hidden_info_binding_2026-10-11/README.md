# The hidden-information binding study (2026-10-11): which hidden fact binds our decisions?

**Status: REGISTERED (the design and every decision rule: [`../../../endstate/design_hidden_info_binding.md`](../../../endstate/design_hidden_info_binding.md)
§12), before any arm of the study was run or read. The cheap parts run in this unit; their reads land in the design
doc's §16 and below, each AFTER the registration commit.**

## In plain language

A value-discrimination pilot (`../value_disc_pilot_2026-10-11/`) found that a search which plays every action out in
the TRUE battle beats our policy by ~7 points of win probability per decision, and that the same search in fair
guessed worlds gains nothing; it blamed the opponent's hidden MOVES. This study tries to reproduce that and to break
it: it hides and reveals each kind of hidden fact (moves, the never-seen Pokémon, items, abilities, spreads) one at a
time in the search's worlds, mixes the true moves into the search's guesses and into the NETWORK's own move belief
with 60–100 % confidence (against a confidently WRONG placebo), checks the world builder for bugs, and checks whether
the effect is a feature of a still-improving 15M policy.

## What runs now (CPU, cores 5–7 + SMT siblings, ≤ 3 workers, `nice -n 19`, `mem_cap`)

| part | design § | files (this folder) | archive |
|---|---|---|---|
| sampler audits A1–A6 | §5 | `audit_*.py`, `audit_*_<cp>.json` | `~/gen3ai_archive/hidden_info_binding/<cp>/audit/` |
| the model-input axis (the true moves injected into the network's move belief) | §7 | `inject.py`, `model_input.py`, `score_model_input.py` (+ test), `unblind_model_input.py`, `arm_key.json`, `inject_teeth_<cp>.json` | `~/gen3ai_archive/hidden_info_binding/<cp>/` |
| descriptive reads D1–D8 of the pilot's rows | §11 | `descriptive_reads.py`, `descriptive_reads.json` | — |
| the pool-vs-prior read D9 | §11 | `poolprior_read.py`, `poolprior_read.json` | — |
| policy drift 1M–15M | §10.1 | `policy_drift_summary.json` | `~/gen3ai_archive/policy_drift/` |
| belief maturity 1M–15M | §10.2 | `belief_maturity.py`, `belief_maturity.json` | — |
| the root facts D2–D4 stratify on (protocol-derived, no outcome) | §11 | `root_facts.py`, `root_facts_<cp>.json` | — |
| POST-HOC breakdowns of D1 (labelled, no verdict) | §16.1 | `posthoc_reads.py`, `posthoc_reads.json` | — |
| the hand evaluator (BUILT; its playout arm runs with the factorial) | §6.5 | `src/main/hidden_info_study/hand_eval.py` (+ test), `hand_eval_pin.py`, `hand_eval_playouts.py`, `hand_eval_arm_key.json`, `hand_eval_pin_check_<cp>.json` | `~/gen3ai_archive/hidden_info_binding/<cp>/hand_eval/` |
| policy drift helpers (the pin checkouts have no Showdown build; team validation redirected to main's byte-identical `validate_team.js`) | §10.1 | `policy_drift_scripts/` | — |

`run_at_pin.sh` is the pilot's runner (a script runs at its checkpoint's OWN pin checkout, CUDA hidden, nice 19,
mem cap, taskset 5-7,13-15). The checkpoints: end2 = `rb_es_end2_s2001` @ `62085750`, prod = `rb_es_prod_s2001` @
`95d014fa`, and the replication network `rb_es_end2_s2002` @ `62085750` (on end2's states).

## Blinding

The model-input arms are written under opaque codes (`arm_key.json`); `score_model_input.py` computes every table by
code and was committed before any arm output was read; `unblind_model_input.py` applies the labels and the
registered rules (design doc §12.8) afterwards.

## Results

*Not yet read* (filled in the commit that reads them).
