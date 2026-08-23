# bridge_corpus — the frozen PER-SIDE / `|request|` regression corpus

Each `*.txt` is a self-contained single-battle bridge-fuzzer repro (a repro dir's
`battle.txt`: `SCEN`/`TEAM`/`INIT`/`CMD`/`SEED`/`CHUNK`/`END` rows), replayed by
`tests/bridge_corpus_test.rs` through the built `bridge_replay --ab` (the per-side/`|request|`
byte differential vs the recorded real `getPlayerStreams`, plus the SEED ANCHOR — each
decision's post-decision engine seed == the recorded omniscient `seedAfter`).

Two fixture classes:

- **Untagged** (`01_*`..`06_*`, `12_*`..`14_*`, both formats) — a CLEAN per-side/request
  battle that MUST replay `ok` (every per-side chunk + `|request|` frame byte-identical,
  seed anchor holds). The round-6 CONTENT-bug guards:
  - `12_morning_sun_success_cg.txt` (bab_7_1) → the B2 fix: a SUCCESSFUL Morning Sun
    weather-heal renders the plain self-target announce (never the did-nothing `[still]`),
    AND the B5-sibling Pursuit-into-a-Pressure-switcher +1 Snatch/Pursuit PP.
  - `13_choice_lock_disabled_cg.txt` (bab_3_24) → the B4 fix: a Choice-Band-locked mon
    (Skarmory gained the CB via Thief mid-turn) shows `disabled:true` on the non-locked
    request moves (the lazy choice lock at request-build).
  - `14_snatch_pressure_pp_cg.txt` (bab_4_16) → the B5 fix: a Snatch steal of a Pressure
    victim's move deducts 2 Snatch PP (the DeductPP Pressure extra).
  - `20_mimic_slot_request_omits_target_cg.txt` (bab_11_17, `--mode random`) →
    `gen3_mimic_request_no_target_v1`: a MIMIC-acquired active slot carries **no `target`
    key at all** (gen3 inherits gen4's Mimic, whose replacement move-slot literal —
    `data/mods/gen4/moves.ts:868` — omits the `target` the base `data/moves.ts` mimic sets,
    so `getMoveRequestData` reads `undefined` and `JSON.stringify` drops the key). The
    mon's other three slots keep theirs, so a blanket omission fails this fixture too.
  - `11_return102_numeric_alias_cg.txt` (bab_0_0) → `gen3_happiness_bp_request_alias_v1`: a
    Return carrier whose request now matches in ALL THREE of the sim's inconsistent alias
    forms — roster moveid `return102`, active id BARE `return`, active display `Return 102`.
    **It was `# ALLOWLIST return102-numeric-alias` until the port learned to emit them**;
    untagging is strictly STRONGER than the tag was, since the tag only required the
    divergence to stay the known one and this requires there to be none.
- **`# ALLOWLIST <reason>` tagged** (`10_*`, `15_*`, `16_*`) — a battle that MUST
  diverge with EXACTLY the tagged `allowlisted` reason:
  - `10_allowlist_curse_target.txt` → `curse-nonghost-target-self-vs-normal`
  - `15_own_typed_hp_roster_curse_cg.txt` (bab_3_19) → `curse-nonghost-target-self-vs-normal`
    — GUARDS the B3 own-typed-HP fix: after the roster serializer resolves the owner's bare
    Hidden Power to the TYPED id (`hiddenpowerdark`), the ONLY residual is the documented
    Curse `target:self`-vs-`normal` deferral; a B3 regression re-introduces a `hiddenpower`
    residual → NOT allowlisted → this fixture FAILS.
  - `16_construction_order_flip_cg.txt` (bab_3_15) → `turn0-construction-speed-tie-order-flip`
    — the B1 NON-mirror construction speed-tie per-side framing ORDER flip (a `-ability`/
    `-weather` permutation at a Tyranitar-213-vs-Suicune-213 tie; seed=None-invisible).

  (The `gender-level-details-construction-draw` deferral is inactive on the pinned-gender
  L100 pool, so it has no fixture yet; the classifier still carries it for randbats/random.)

To ADD a fixture: run `node harness/bridge_ab_fuzz.js --mode pool --format {gen3customgame|gen3ou}`,
take a repro's `battle.txt` (or extract a clean battle from `chunks/`), and drop it here. A clean
battle stays untagged; a documented-deferral battle gets a `# ALLOWLIST <reason>` header line.
The `ab_fuzz_out*` run dirs are gitignored — never commit raw run output.
