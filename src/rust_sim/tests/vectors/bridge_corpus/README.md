# bridge_corpus — the frozen PER-SIDE / `|request|` regression corpus

Each `*.txt` is a self-contained single-battle bridge-fuzzer repro (a repro dir's
`battle.txt`: `SCEN`/`TEAM`/`INIT`/`CMD`/`SEED`/`CHUNK`/`END` rows), replayed by
`tests/bridge_corpus_test.rs` through the built `bridge_replay --ab` (the per-side/`|request|`
byte differential vs the recorded real `getPlayerStreams`, plus the SEED ANCHOR — each
decision's post-decision engine seed == the recorded omniscient `seedAfter`).

Two fixture classes — **14 clean + 3 allowlisted** today:

- **Untagged** (`01_*`..`06_*`, `10_*`..`15_*`, `19_*`, `20_*`, both formats) — a CLEAN
  per-side/request battle that MUST replay `ok` (every per-side chunk + `|request|` frame
  byte-identical, seed anchor holds). The round-6 CONTENT-bug guards:
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
  - `10_allowlist_curse_target.txt` → `gen3_bridge_curse_request_target_v1`: a NON-GHOST
    Curse's `|request|` move target is the sim's runtime `nonGhostTarget` `"self"`, not the
    base-dex `"normal"`. **The file NAME records the class it used to sit in**; a regression
    re-introduces the `curse-nonghost-target-self-vs-normal` divergence → NOT allowlisted →
    this fixture FAILS.
  - `15_own_typed_hp_roster_curse_cg.txt` (bab_3_19) → the B3 own-typed-HP fixes
    (`gen3_own_typed_hp_request_roster_v1` + `gen3_own_typed_hp_active_request_v1`) on top of
    the Curse-target one: the owner's own bare-stored Hidden Power resolves to the TYPED form
    in BOTH the roster (`hiddenpowerdark`) AND the active moves (`"Hidden Power Dark 70"`). A
    regression of ANY of the three re-introduces a request divergence → this fixture FAILS.
  - `06_anchor_clean_phaze_bab_0_16_cg.txt` / `19_anchor_clean_double_switch_ou.txt` → the A2
    SEED-ANCHOR fixtures (`gen3_perside_seed_anchor_makerequest_align_v1` /
    `gen3_perside_seed_anchor_subsequence_v1`): a phaze-DRAG battle and a Zapdos-Pressure
    mirror with a SEQUENTIAL double-forced-switch, where the port records an EXTRA
    makeRequest checkpoint the omniscient per-decision capture collapses. Both boards are
    byte-CLEAN, so the anchor must absorb the extra boundary → `ok`; reverting either anchor
    fix re-reports `kind=seed`.
- **`# ALLOWLIST <reason>` tagged** (`16_*`..`18_*`) — a battle that MUST
  diverge with EXACTLY the tagged `allowlisted` reason:
  - `16_construction_order_flip_cg.txt` (bab_3_15) → `turn0-construction-speed-tie-order-flip`
    — the B1 NON-mirror construction speed-tie per-side framing ORDER flip (a `-ability`/
    `-weather` permutation at a Tyranitar-213-vs-Suicune-213 tie; seed=None-invisible).
  - `17_perside_construction_mirror_flip.txt` (bab_7_13, gen3ou) →
    `perside-construction-speed-tie-mirror-of-flip` — form (a): the turn-0 construction
    speed-tie SAME-SPECIES MIRROR weather-`[of]` flip on the p1 per-side stream (a
    Tyranitar-Tyranitar Sand Stream mirror), the per-side analog of `ab_replay.rs`'s A1 key.
  - `18_perside_construction_mirror_permutation.txt` (bab_6_14, gen3customgame) →
    `perside-construction-speed-tie-mirror-of-flip` — form (b): the same-species MIRROR
    Intimidate BLOCK PERMUTATION, where both `-ability|…|Intimidate|boost` lines reorder
    TOGETHER with their paired `-unboost|…|atk|1` — an IDENTICAL multiset that B1's clause-3
    (`-unboost` rejected) cannot catch.

  (The `gender-level-details-construction-draw` deferral is inactive on the pinned-gender
  L100 pool, so it has no fixture yet; the classifier still carries it for randbats/random.)

To ADD a fixture: run `node harness/bridge_ab_fuzz.js --mode pool --format {gen3customgame|gen3ou}`,
take a repro's `battle.txt` (or extract a clean battle from `chunks/`), and drop it here. A clean
battle stays untagged; a documented-deferral battle gets a `# ALLOWLIST <reason>` header line.
The `ab_fuzz_out*` run dirs are gitignored — never commit raw run output.
