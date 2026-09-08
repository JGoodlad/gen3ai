# The data-driven mechanic classes — the wired classes and the per-class roadmap

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

Items and abilities are not hand-modelled one id at a time: every gen3-resolved entry is
classified into a mechanic CLASS with machine-readable parameters, and ONE generic engine path
serves each class. This document holds every wired class (Phase 1 stat/BP item folds, Phase 2
ability DMG_MOD, Phase 3 accuracy pipeline, batches 1-4, STATUS_IMMUNE, SWITCH_OUT,
TYPE-INTERACTION) with its parameters, draw model and validation, plus the per-class roadmap and
the two real misses the handler audit's first run surfaced. The leaf keeps the framework, the
MOD-CHAIN LAW, the drift gate and the audit's invocation.

---

- **The wired classes (Phase 1) — the STAT/BP-MODIFIER item family, all folds probe-settled:**
  - **TYPE_BOOST (24)** — 18 stat-fold members (`onModifyAtk/SpA chainModify` ×1.1 + Sea Incense
    ×1.05 → the offensive-stat chain), the 2 gen2 bows (`return basePower * 1.1` — a DIRECT
    float that REPLACES the event relayVar; the non-integer product SKIPS runEvent's final chain
    modifier and `clampIntRange` floors — mirrored exactly in `get_base_damage`), and the 4
    incenses (`chainModify([4915,4096])` ≈ **×1.2, NOT the assumed ×1.1** — the probe's headline
    surprise) at the BASE-POWER chain, ONE accumulated 4096 modifier shared with Thick Fat.
  - **SPECIES_STAT (6)** — Thick Club (Atk ×2 Cubone/Marowak), gen3 Light Ball (SpA-ONLY ×2
    Pikachu), DeepSeaTooth (SpA ×2 Clamperl), DeepSeaScale (SpD ×2 Clamperl — the first
    DEFENDER-side stat fold: `resolve_def_stat_mods` → the `ModifyDef/ModifySpD` chain, after
    the boost table, BEFORE the gen≤4 Explosion Def-halve), Metal Powder (Def ×2 untransformed
    Ditto), Soul Dew (SpA+SpD ×1.5 Lati@s — both directions).
  - **CHOICE** — Choice Band (`choice: true` + `statMods.atk [3,2]`; the lock was already
    modeled via `choice_locked_move`).
  `turn.rs::resolve_atk_stat_mods` / `resolve_def_stat_mods` / `resolve_bp_mods` are pure
  dex-data lookups — the hardcoded item match-arm is GONE, so the e2e `MODELED_ITEMS` (which
  gained the 6 species items) can never drift from the engine again for these classes. The
  confusion self-hit resolves the SAME helpers (a Thick Club Marowak's self-hit uses ×2 Atk, a
  Metal Powder Ditto's is halved by its own Def — the full-`getDamage` semantics the CB e2e_194
  fix established). All folds are DRAW-FREE.
- **Validation:** `harness/gen_damage_golden.js` grew 17 EXACT max-roll probes (48 scenarios
  total; 3 columns appended — atk_species/def_species/def_item — pre-existing lines
  prefix-identical) → `tests/damage_test.rs` (whose item mirrors are now dex-data-driven too);
  the CLASS-SWEEP golden `harness/gen_item_mods_golden.js` → **`tests/item_mods_test.rs`** (33
  scenarios × 30 seeds = 990 battles to game-end, 2664 per-decision STATE+HP+SEED assertions,
  1398 boosted-hit rows, ≥10 boosted hits enforced per member, matching + wrong-type/
  wrong-species controls; byte-reproducible); PERTURBATION-PROVEN (incense→×1.1, a leaked gen4
  Light-Ball Atk half, a removed species gate, a def-mult drift — each fails the gate; restores
  byte-identical); 6 revert-verified pins `IM1`-`IM6` in `tests/regression_test.rs`. The e2e
  regen after the `MODELED_ITEMS` growth left the golden BYTE-UNCHANGED (no pool team became
  newly filter-clean) — STRICT 220/220 stands.
- **The wired classes (Phase 2) — the ABILITY DMG_MOD family, all folds probe-settled against
  the RESOLVED gen3 dist (`gen3_item_mechanics_v1` ability side, `dex/abilities.rs::AbilityData.dmg_mod`):**
  - **PINCH (4)** — Torrent/Blaze/Overgrow/Swarm: an `onBasePower chainModify(1.5)` for the
    ability's type (Water/Fire/Grass/Bug) when the user is at `hp <= maxhp/3` — bit-exactly the
    integer `3*hp <= maxhp` (probe-verified at the maxhp=341 float boundary). A BP-chain member
    (`resolve_bp_mods`, joins the incense/Thick-Fat accumulate-once).
  - **Atk uncond (2)** — Huge/Pure Power: `onModifyAtk chainModify(2)`, PHYSICAL only (ModifyAtk
    touches only the Atk stat; a special move is un-boosted). An `AtkStatMod` in the ModifyAtk
    chain (`resolve_atk_stat_mods`), so Guts+CB stacks to ONE ×2.25 chain (not two rounds).
  - **Guts (Atk ×1.5 whenStatused)** — `onModifyAtk chainModify(1.5)` when the user has ANY major
    status, PLUS the physical burn-halve is SUPPRESSED (the port's existing `Combatant::has_guts`
    skip in `modify_damage`). PROBED: a burned Guts mon hits at full ×1.5 (=×1.497 realized), NOT
    ×0.75 — the ×1.5 stat fold composes with the burn-skip.
  - **Marvel Scale (Def ×1.5 whenStatused)** — `onModifyDef chainModify(1.5)` while the DEFENDER
    has a major status (the physical Def only), a `def_stat_mods` member (`resolve_def_stat_mods`).
  Hustle stays DATA-ONLY (`direct`, `this.modify(atk,1.5)` — its Atk half ships WITH the accuracy
  pipeline's ×0.8 acc side, never alone). Thick Fat keeps its dedicated `defender_thick_fat` hook
  (its `sourceBasePower` fold is a DEFENDER handler on the ATTACKER's BP). The confusion self-hit
  resolves the SAME helpers with the mon's own status on both sides (a burned Guts mon's self-hit
  is ×1.5 + burn-suppressed; a statused Marvel mon's self-hit is reduced — probe-verified). All
  folds are DRAW-FREE.
- **Validation (Phase 2):** `dex/abilities.rs` grew an `AbilityData` (replacing the abilities'
  `NamedNum`) that parses the `dmgMod` fields (`ability_dmg_mod_fields_parse`). The damage golden
  `harness/gen_damage_golden.js` grew 15 EXACT max-roll probes (63 scenarios total; a constructed
  pinch-HP/status hook + 4 columns appended — `atk_hp`/`atk_maxhp`/`def_status`/`def_ability` —
  pre-existing 28 indices unchanged) → `tests/damage_test.rs`; the CLASS-SWEEP golden
  `harness/gen_ability_dmgmod_golden.js` → **`tests/ability_dmgmod_test.rs`** (11 scenarios × 30
  seeds ≈ 330 decisive battles to game-end, per-decision STATE+HP+SEED, ≥10 boosted hits enforced
  per member, incl. wrong-type-pinch + unstatused-Guts controls; byte-reproducible). REVERT-PROVEN
  (the golden test catches every engine-fold revert — pinch/Huge/Pure/Guts/Marvel each diverges a
  battle); 5 revert-verified pins `AB1`-`AB5` in `tests/regression_test.rs` (AB1 Guts+burn, AB2 the
  pinch threshold boundary, AB3 Marvel def-side, AB4 Huge ×2, AB5 the status gate). **e2e admission
  DONE (`gen3_sun_freeze_immunity_v1`):** `torrent/blaze/overgrow/swarm/hugepower/purepower/guts/
  marvelscale` are now in `MODELED_ABILITIES` — the filter-clean pool grew **22 → 151 / 719** (the
  biggest single admission lever; the DMG_MOD gaps VANISHED from the taxonomy's top-gaps list), STRICT
  `filtered_diverged == 0` over 220 battles / 9963 decisions, byte-reproducible at the committed knobs.
  The admission was previously gated on ONE newly-admitted battle (e2e_86) that diverged on a
  PRE-EXISTING, non-DMG_MOD gap — the **SUN → freeze immunity** (`sunnyday.onImmunity('frz')` blocks a
  freeze while the field is Sun; the port used to freeze anyway — the A/B "ice-freeze cluster"). That
  is now FIXED (`turn.rs::try_set_status` sun-freeze gate, pinned FZ1). Admitting the abilities ALSO
  surfaced ONE cascade in an already-modeled mechanic — a Gengar packed with the move ALIAS `wisp`
  (Will-O-Wisp), which the port used to NO-OP (the dex read only canonical ids) — FIXED via
  `gen3_move_aliases.json` + Rust-dex alias resolution (pinned FZ2). No 219/220 or per-battle exclude
  was needed — the enlarged corpus is a CLEAN strict pass.
- **The wired classes (Phase 3) — the ACCURACY pipeline, all folds probe-settled against the
  RESOLVED gen3 dist (`gen3_accuracy_pipeline_v1`; `dex/accmod.rs::AccMod`, consumed by
  `turn.rs::effective_accuracy`/`roll_accuracy`):** the to-hit roll is now
  `effAcc = move.accuracy × the acc/eva STAGE TABLE × the accMod item/ability handlers`, then the
  ONE `random(100) < effAcc` draw. **DRAW-RELEVANT** (unlike the P1/P2 stat/BP folds): a wrong effAcc
  flips a hit↔miss → the crit/damage draws follow only on a hit → the seed desyncs, so the math AND
  draw-order are bit-for-bit. Members (all from the RESOLVED dist — the mod-chain law; the base `.ts`
  shapes differ, e.g. base Bright Powder is `chainModify([3686,4096])` but the gen3 mod REWRITES it to
  a DIRECT `accuracy * 0.9`):
  - **acc/eva STAGE TABLE** — gen3 `[3/3,4/3,5/3,6/3,7/3,8/3,9/3]`, applied inline BEFORE ModifyAccuracy:
    the attacker's accuracy stage (`boosts[5]`: `acc *= table[+s]` / `/= table[-s]`), the defender's
    evasion stage (`boosts[6]`: `acc /= table[+s]` for +, `*= table[-s]` for −). The result stays a
    RAW f64 into the comparison (`random(100)` is an integer 0..99 → `int < effAcc_f64` matches JS).
    (A foe accuracy-drop reaches `boosts[5]` via a modeled move's secondary — e.g. Mud-Slap/Muddy
    Water, the fuzzer's cluster; the extractor already emits accuracy in `secondaryBoosts`.)
  - **ACCURACY_ITEM (2)** — Bright Powder ×0.9 / Lax Incense ×0.95 (`AccOp::Multiply`, DEFENDER-side,
    a DIRECT float that mutates relayVar unconditionally).
  - **ACCURACY ability (3)** — Compound Eyes ×1.3 (attacker), Sand Veil ×0.8-in-sand (defender) +
    its `onImmunity('sandstorm')` sand-chip immunity (the ONLY gen3 weather-chip onImmunity, folded
    into `weather_immune`), Hustle ×3277/4096 for a physical-TYPE move (attacker — the gen3-mod gates
    on `move.type ∈ physicalTypes`, NOT category). All three are `AccOp::Chain` — accumulated into ONE
    4096 modifier applied at the END of runEvent via `modify(acc, modifier)` BUT ONLY when `acc` is a
    NON-NEGATIVE INTEGER (`relayVar === Math.abs(Math.floor(relayVar))`) — so a stage or a direct
    multiply having made `acc` a float SKIPS every chain member (the integer-guard, probe
    `harness/probe_accuracy_intguard.js`, mirrored in `accuracy_chain_modify` + its caller).
  **HUSTLE ships FULLY this phase** — its Atk ×1.5 (`dmgMod`, a DIRECT `this.modify(atk,1.5)` applied
  as a separate pre-chain `modify` in `build_damage_context.atk_direct_modify`, NOT a chainModify
  accumulation) pairs with its acc ×0.8; Hustle is OFF the DATA-ONLY/deferred list. All stage/mod math
  is DRAW-NEUTRAL (still exactly one accuracy draw); the empty/no-stage/no-mod path is BYTE-IDENTICAL
  (`acc` stays the integer `move.accuracy` → `random(100) < accuracy` == the pre-pipeline
  `random_chance(accuracy,100)`).
- **Validation (Phase 3):** `dump_gen3_mechanics.js` grew an `accMod` extraction (the RESOLVED
  onModifyAccuracy/onSourceModifyAccuracy handler → `{op, mod, side, weather?, physicalTypesOnly?}`);
  the `--check` drift gate now pins `accMod` on both items + abilities (132 items + 76 abilities match
  the dist). `dex/accmod.rs` parses it into `AccMod` (`item_acc_mod_fields_parse` /
  `ability_acc_mod_fields_parse`); `turn.rs::effective_accuracy_matches_sim_probe` pins the exact effAcc
  math bit-for-bit vs the sim-captured probe (stage table, each accMod member, the integer-guard). The
  CLASS-SWEEP golden `harness/gen_accuracy_golden.js` → **`tests/accuracy_test.rs`** (7 scenarios × 30
  seeds — the acc-stage fold via Muddy Water, each accMod member, a no-mod control; per-decision
  STATE+HP+**SEED** to game-end, ≥10 miss/accMod rows enforced per member, the control fires 0 accMod;
  byte-reproducible). REVERT-PROVEN (each fold's revert flips a roll → the golden test diverges); 5
  revert-verified pins `AC1`-`AC5` in `tests/regression_test.rs` (AC1 the acc-stage flip — the
  fuzzer's Sand-Attack/Mud-Slap cluster, AC2 Compound Eyes, AC3 Bright Powder, AC4 Hustle Atk×1.5+acc,
  AC5 Sand Veil's sandstorm-chip immunity).
  The e2e regen after adding `brightpowder`/`laxincense` (items) + `compoundeyes`/`sandveil`/`hustle`
  (abilities) to the `MODELED_*` sets grew the filter-clean pool (see the e2e note); STRICT
  `filtered_diverged == 0` over 220 battles, byte-reproducible at the committed knobs.
- **The per-class roadmap (execute against the inventory's class map):**
  1. **ACCURACY pipeline** — **WIRED (Phase 3, see above).** acc/eva stages + the ACCURACY_ITEM class
     (Bright Powder / Lax Incense) + Compound Eyes / Sand Veil + Hustle's ×0.8 physical-acc side (+ its
     Atk ×1.5); the fuzzer's acc-stage cluster is fixed.
  2. **Ability DMG_MOD** — **WIRED (Phase 2, see below).** The pinch family
     Torrent/Blaze/Overgrow/Swarm (BP chain ×1.5 at hp≤⅓, `3·hp<=maxhp` exact), Huge/Pure
     Power (Atk ×2), Guts (Atk ×1.5 statused + the burn-halve suppressed), Marvel Scale
     (Def ×1.5 while the DEFENDER is statused); Hustle now ships fully with the accuracy phase.
  3. **Berries — WIRED + E2E-ADMITTED (BATCH 3, `gen3_berry_trace_shedskin_v1`, 2026-07-07;
     admission + regen 2026-07-08 — see EDGE_CASES.md's batch-3 section for the
     full record).** ONE eatItem consumption mechanism (`MonState::item` — the CURRENT item;
     eaten → NONE permanently) + 22 data-driven `berryEffect` rows (`gen3_items.json`, extractor
     + `--check` gate, obs-neutral): CURE (7 — the Update-site eat, BEFORE the holder's move;
     LUM immediate-in-setStatus after Synchronize, incl. LumRest), HEAL (7 — residual order 10
     subOrder 4 = the LEFTOVERS slot, `2*hp<=maxhp` exact; oran 10 / sitrus 30 / the Figy family
     floor(maxhp/8) + nature-gated confusion random(2,6)), PINCH (7 — `4*hp<=maxhp`; +1 boosts;
     Starf's n≥1 `sample` +2; Lansat's focusenergy crit+2), PP (leppa +10 capped). Probe-settled
     (`probe_berry_rng.js`/`probe_berry_sub_tie_rng.js`: the eat is DRAW-FREE; a berry-vs-Leftovers
     equal-speed mirror draws IDENTICALLY; subs don't trigger it; a KO'd holder never eats;
     **`probe_berry_threshold_boundary.js`: the sim eats AT exact equality** — an EVEN-maxhp board
     [Vaporeon 400] landing on hp == maxhp/2 / maxhp/4 exactly EATS, one-HP-above does not, so
     `<=` is the probe-settled boundary, pinned by BR6 [the prior odd-maxhp boards left `<=`-vs-`<`
     unfalsifiable — the closed reviewer finding]);
     validated by `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` (1280 battles,
     per-decision STATE+STATUS+**ITEM**+BOOSTS+SEED, byte-reproducible) + pins BR1-BR3 + BR6.
     **TRACE + SHED SKIN ship in the same batch** (pins BR4/BR5): trace = the gen3-resolved
     onStart n=1 `randomFoe` sample + a LIVE current-ability copy (`MonState::ability`; no
     copied-onStart in gen3; switch-out reverts; lead-trace draw-free at the seeded start;
     fail-loud `TRACE_COPYABLE` guard — kept in LOCKSTEP with the e2e MODELED∪NOOP allow-lists),
     shed skin = residual order 10 subOrder 3
     `randomChance(33,100)` per STATUSED residual (cure before the DoT; handler gathered
     unconditionally for the tie-shuffle). **The e2e admission grew the filter-clean pool
     585 → 712 / 719** (the biggest admission since Natural Cure) — a CLEAN STRICT pass
     first-try, NO new engine bug, leech finally exercised (354 decisions, was 0).
  4. **PROC_ITEM — WIRED (BATCH 4, see the batch-4 note).** King's Rock (the appended trailing
     `{chance:10, flinch}` secondary for the LISTED moves — `ItemData::flinch_secondary`, the
     execution-derived 130-id list in `gen3_items.json`; [own secondary]→[KR]→[contact proc] order,
     Serene Grace ×2, Shield Dust filters, drawn-not-applied behind a sub, Seismic Toss/Struggle
     proc too — probes `probe_kingsrock_rng.js` + `probe_kingsrock_order_rng.js`) + Focus Band
     (`ItemData::survive_lethal` — `focus_band_damage` at EVERY Damage-event site: move hits,
     burn/sand chips, the leech drain, Spikes, Struggle/Rough-Skin recoil, confusion self-hits
     [effectType Move → CAN survive]; NOT sub-absorbed hits — probes `probe_focusband_rng.js` +
     `probe_focusband_confusion_rng.js`); Quick Claw was already modeled.
  5. **CRIT_ITEM — WIRED (`gen3_crit_item_v1`).** Scope Lens (+1 unconditional) / Lucky Punch
     (+2 Chansey) / Stick (+2 Farfetch'd) fold `onModifyCritRatio critRatio + N` into
     `turn.rs::effective_crit_ratio` (the Focus Energy precedent — DRAW-FREE, only the `CRIT_MULT`
     denominator index shifts; the crit `randomChance(1, denom)` draw COUNT is unchanged). Data:
     the `critBoost {boost, onlySpecies}` field in `gen3_items.json` (`ItemData.crit_boost`,
     extractor + `dump_gen3_mechanics.js --check` drift gate). Species-gated via `user.species.id`.
     `leek` is the gen8 rename — NOT gen3-legal. 0 team-carry (Farfetch'd/Chansey aren't gen3 OU),
     so admitting `scopelens`/`luckypunch`/`stick` to `MODELED_ITEMS` is byte-neutral on the e2e
     sample; proven by the CI1/CI2 pins (Stick crits vs a no-item control at the same seed → same
     seedAfter, the draw-free proof). Still UNMAPPED: DRAIN_ITEM (Shell Bell), BOOST_RESTORE (White
     Herb), CURE_ITEM (Mental Herb), SPEED_MOD (Macho Brace), TAKE_ITEM_GUARD (Mail).
  6. **Ability classes beyond DMG_MOD** — **SWITCH_OUT (Natural Cure) ✅ DONE** (`gen3_natural_cure_v1`,
     2026-07-06 — the BIGGEST admission lever, 151 → 449) + **STATUS_IMMUNE ✅ DONE** (`gen3_status_immune_v1`,
     2026-07-06, below — the #2 gap, immunity=97) + **BATCH-1 ✅ DONE** (`gen3_ability_batch1_v1`, 2026-07-07,
     below — **CRIT_IMMUNE** [shellarmor/battlearmor], **WEATHER_SPEED** [chlorophyll/swiftswim],
     **WEATHER_NEGATE** [cloudnine/airlock], **RESIDUAL** [speedboost/raindish]; 525 → **571**, shellarmor the
     lever) + **BATCH-2 ✅ DONE** (`gen3_ability_batch2_v1`, 2026-07-07, below — the DRAW-BEARING
     "reactive" classes: **CONTACT_PROC** [static/poisonpoint/flamebody/effectspore, `randomChance(1,3)` or
     Effect Spore's `random(10)`+`sample(3)` → status the ATTACKER, AFTER the move secondary] +
     **CONTACT-recoil** [roughskin, maxhp/16 draw-free] + **BLOCK** [soundproof / damp / suctioncups] +
     **SYNCHRONIZE** [reflect a foe status to the source]; 571 → **585**, synchronize [the #1 taxonomy gap]
     + effectspore the levers) + **BATCH-3 ✅ DONE** (`gen3_berry_trace_shedskin_v1`, 2026-07-07/08,
     roadmap item 3 above — **TRACE** [was the #1 gap] + **SHED_SKIN** + the 22 berries, all wired,
     goldened, pinned BR1-BR6, AND e2e-admitted: 585 → **712 / 719**, the biggest admission since
     Natural Cure) + **BATCH-4 ✅ DONE** (`gen3_ability_batch4_v1`, 2026-07-08, the FINAL mechanics
     tail — **TRUANT** [was the last team-carry gap, =4], **INNER FOCUS** [=2], **SHADOW_TAG**,
     **CUTE CHARM + the ATTRACT volatile**, **COLOR CHANGE** [the `MonState::types_override` thread
     through the ONE `mon_types` choke point], and the PROC_ITEM pair **KING'S ROCK** [the appended
     trailing 10% flinch secondary over the execution-derived 130-move list] + **FOCUS BAND** [the
     onDamage `randomChance(1,10)` on EVERY Damage event into the holder; survive-at-1 on a lethal
     MOVE hit] — see the batch-4 note below). **FORECAST — the LAST member — is DONE**
     (`gen3_forecast_v1`, ROUND 35 above): the forme+TYPE swap at every WeatherChange site
     (incl. the previously-missing UNCONDITIONAL expiry draw — the T1 8-vs-7 fix), the entrant
     `onStart`, the start window, the silent `clearVolatile` revert, and the
     reporting surfaces reading the construction `base_species_id` (the forme name appears on
     the wire ONLY in the `|-formechange|` line). The old construction FAIL-LOUD
     (`gen3_forecast_failloud_v1`) is RETIRED — pinned by its negative control
     `forecast_castform_builds_now_that_forecast_is_modeled` — and
     `REJECT_ABILITIES`/`REJECT_SPECIES` are EMPTY (kept as the seam for the next ability
     deferral): gen3-randbats Castform teams now reach the port. The ability class census is
     therefore COMPLETE: every gen-3 ability is modeled or verified-no-op, none fail-loud.
  - **STATUS_IMMUNE ability — DONE as a DATA-DRIVEN class** (`gen3_status_immune_v1`, 2026-07-06). The gen-3
    abilities that grant immunity to a specific MAJOR status: **Limber** (par) / **Insomnia** + **Vital
    Spirit** (slp) / **Immunity** (psn,tox) / **Water Veil** (brn) block via `onSetStatus`; **Magma Armor**
    (frz) blocks via `onImmunity` (BEFORE the SetStatus event). Own Tempo (confusion) + Oblivious (attract)
    block a VOLATILE via `onTryAddVolatile`, NOT a major status → NOT members (Leaf Guard is num 102 = NOT
    gen-3). DATA-DRIVEN: the extractor emits `statusImmune {statuses, phase}` into `gen3_abilities.json`
    (`_GEN3_ABILITY_MECHANICS`, drift-gated by `dump_gen3_mechanics.js --check` which DERIVES it from the
    resolved `onSetStatus`/`onImmunity` handlers; obs-neutral) → `dex/abilities.rs::AbilityData.status_immune`
    (`StatusImmune {statuses, phase: SetStatus|Immunity}`) → `turn.rs::try_set_status` reads it (the
    `Immunity` phase gates BEFORE `set_status_event_shuffle`, the `SetStatus` phase AFTER). **THE
    PROBE-SETTLED DRAW MODEL** (`harness/probe_statusimmune_{rng,setstatus_event,shuffle_size,
    magmaarmor,enumerate}.js`): DRAW-FREE in gen3customgame (the ability is the SetStatus event's only handler
    → size-1 → NO shuffle; Magma Armor blocks before the event) — so admission is SEED-CLEAN. In gen3ou an
    `onSetStatus`-phase ability adds a 3rd SetStatus handler, but it sorts into its OWN speed group (defined
    `speed` beats the clauses' `undefined`), leaving the 2 clauses a SIZE-2 tie → `shuffle(list,1,3)` draws
    EXACTLY ONE `random`, IDENTICAL to the control's `shuffle(list,0,2)` — so the draw COUNT is UNCHANGED
    (this REFUTED + REMOVED the old "size-3 shuffle" fail-loud panic). It was the **#2 e2e team-carry gap**
    (immunity=97); admitting the 6 members (+ moving `insomnia`/`vitalspirit` out of `NOOP_ABILITIES` —
    they genuinely block sleep) grew the filter-clean pool **449 → 525 / 719** (immunity=97, the #2 gap). The
    enlarged corpus surfaced + FIXED ONE real engine bug (NOT the STATUS_IMMUNE class) — the **EMPTY NATURE**
    (e2e_8/e2e_73 carry a Suicune with an OMITTED nature field, which the sim treats as NEUTRAL/Serious but the
    port PANICKED on; `stats.rs::compute_stats` now computes the neutral all-1.0 multipliers for an empty
    nature, VERIFIED vs the sim — pinned `empty_nature_computes_the_neutral_stats`). STRICT 220/220 clean
    (`filtered_diverged == 0`, 11651 decisions), byte-reproducible; `immunity` DROPPED OFF the taxonomy. Golden
    `gen_statusimmune_golden.js` → `statusimmune_test.rs` (480 game-end battles, the block observable on the
    active-status timeline + a stable-md5 byte-reproducibility gate); pins `limber_blocks_paralysis_draw_free`
    / `insomnia_blocks_sleep_draw_free` / `magma_armor_blocks_freeze` / `immunity_blocks_tox_but_not_burn`
    (SI1-SI4) in `regression_test.rs`; probes `probe_statusimmune_*.js`.
  - **BATCH-1 DRAW-FREE / STRUCTURAL classes — DONE** (`gen3_ability_batch1_v1`, 2026-07-07). FOUR ability
    classes, each DRAW-FREE (or draw-neutral) + validated by the class-sweep golden
    `harness/gen_ability_batch1_golden.js` → `tests/ability_batch1_test.rs` (**300 game-end battles, 941
    per-decision STATE+HP+SEED rows + 1582 spe-boost assertions, byte-for-byte**) + the **B1-B4b** revert-verified
    pins in `regression_test.rs` (ground truth `harness/probe_ability_batch1_regression_rng.js`):
    - **CRIT_IMMUNE** (Shell Armor / Battle Armor) — a hit into the holder NEVER crits: the crit
      `randomChance` is DRAWN normally (draw-count unchanged) then `runEvent('CriticalHit')` reads the
      defender's `onCriticalHit=false` and OVERRIDES the crit to false (`turn.rs`, after the crit roll,
      re-resolve damage with `crit=false`). DRAW-FREE — `probe_critimmune_rng.js` (Slash into Battle Armor:
      IDENTICAL draw count vs a Sturdy control, 0 crits vs the control's ~1/8). B1
      `battle_armor_prevents_the_crit_but_draws_the_roll`.
    - **WEATHER_SPEED** (Chlorophyll / Swift Swim) — `onModifySpe chainModify(2)` in EFFECTIVE sun / rain,
      folded into `effective_speed`'s ModifySpe chain (accumulated with paralysis ×0.25 into ONE 4096
      modifier), so the CACHED speed the eachEvent tie-shuffles + the action-order sort read includes the ×2.
      A slow Chlorophyll mon that ties/overtakes the foe at ×2 flips the first-mover. B2
      `chlorophyll_speed_doubles_and_flips_the_first_mover_in_sun`.
    - **WEATHER_NEGATE** (Cloud Nine / Air Lock) — `effective_weather()` returns None while a negater is
      active (the sim's `suppressingWeather()`), so the sand/hail chip is not scheduled AND the weather-speed
      ×2 doesn't apply; the RAW `field.weather` persists (upkeep/counter). B3 `cloud_nine_suppresses_the_sandstorm_chip`.
    - **RESIDUAL** (Speed Boost / Rain Dish) — `ResidualAction::SpeedBoost`/`RainDish` at residualOrder 10
      subOrder 3 (BEFORE Leftovers sub 4), DRAW-FREE: Speed Boost `+1 spe` stage per active turn
      (`if (pokemon.activeTurns)` — a switch-in skips its entry turn; the boost updates the stage but NOT
      `cached_speed`, so it takes effect NEXT turn), Rain Dish `+maxhp/16` heal in EFFECTIVE rain. B4
      `speed_boost_raises_the_spe_stage_by_one_each_active_turn` + B4b `rain_dish_heals_each_end_of_turn_in_rain`.
    **The STEP-1 sun/rain `eachEvent('Weather')` fix shipped with this batch** (see EDGE_CASES + the
    `run_residuals` note): gen3 sun/rain fire the end-of-turn `eachEvent('Weather')` tie-shuffle
    UNCONDITIONALLY (the port used to gate it on Sand|Hail), so a WEATHER-TURN speed tie under sun/rain drew
    one fewer call → a 1-draw desync. FIXED (schedule the field weather-residual off RAW weather for sun/rain,
    off `effective_weather()` for sand/hail — a negater silences sand/hail but NOT sun/rain, probe-verified);
    pinned `sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed`. **e2e admission** grew the
    filter-clean pool **525 → 571 / 719** (shellarmor the big lever; STRICT `filtered_diverged == 0` over 220
    battles / 11630 decisions, a CLEAN pass first-try — NO new engine bug). The class-(a) NO-OPS
    `plus`/`minus`/`lightningrod`/`stickyhold` are admitted (the 2026-07 no-op verification tested
    them PARTNER-LESS vs an Insomnia control, `probe_ability_batch1_noop_verify.js` — which missed
    the CROSS-FIELD Plus↔Minus pairing; **`plus`/`minus` are now MODELED**, `gen3_plus_minus_v1`,
    probe `probe_plus_minus_gen3.js`; `lightningrod`/`stickyhold` remain true no-ops); **FORECAST is DEFERRED** (a Castform forme+TYPE change under
    rain/sun/hail — the probe diverges — not a no-op; now **FAIL-LOUD** in `state::from_set` + fuzz-rejected —
    see the deferred fail-loud section below). DEFERRED to batch 2: the DRAW-BEARING procs
    (static/poisonpoint/flamebody/cutecharm/effectspore/synchronize/shedskin/trace/shadowtag/roughskin/colorchange).
  - **BATCH-2 DRAW-BEARING "reactive" classes + block tail — DONE** (`gen3_ability_batch2_v1`, 2026-07-07).
    The draw-bearing ability procs the batch-1 note deferred, each PROBE-settled (the sim is the only oracle;
    `harness/probe_contact_proc_{rng,lands}.js` + `probe_effectspore_sample.js` + `probe_block_abilities_rng.js`
    + `probe_synchronize_rng.js`) + validated by the class-sweep golden `gen_ability_batch2_golden.js` →
    `tests/ability_batch2_test.rs` (**960 game-end battles, per-decision STATE+HP+STATUS+SEED, 3250 seed +
    5540 status assertions, byte-for-byte**) + the **B2-1..B2-7** revert-verified pins in `regression_test.rs`
    (ground truth `harness/probe_ability_batch2_regression_rng.js`):
    - **CONTACT_PROC** (Static par / Poison Point psn / Flame Body brn / Effect Spore slp|par|psn) — a
      DATA-DRIVEN `onDamagingHit` (`AbilityData.contact_proc`, `{statuses, chance, sample}`, extracted from the
      resolved dist): when the HOLDER is hit by a **CONTACT** move (`MoveData::contact`, the new `flags.contact`
      field) that dealt damage, it draws `randomChance(chance)` and (on a pass) inflicts a status ON THE ATTACKER.
      **THE DRAW-MODEL CRUX** (probe-settled): the proc's `randomChance` draws INSIDE `runEvent('DamagingHit')`
      (gen<5, battle-actions.ts:982) which the sim fires **AFTER** the move's OWN `secondaries()` (line 957) — so
      the ORDER in the draw stream is `[move secondary random(100)]` THEN `[contact-proc randomChance]`. Static/PP/
      FB roll ONE `randomChance(1,3)` → the single status; **Effect Spore** rolls `randomChance(1,10)` then, on a
      pass, ONE `sample(["slp","par","psn"])` (a `random(3)`) → the sampled status (the NESTED draw). The proc
      draws even **behind a Substitute** (the sub-absorbed target is still a DamagingHit target) and **on a KO**;
      the status lands on the ATTACKER with the gen-3 type/ability/already-statused gates (in gen3ou the reflected
      `trySetStatus` draws the SetStatus 2-clause shuffle — draw-free in the e2e customgame). Wired in
      `turn.rs::apply_contact_proc`, called from `run_move`'s landed-hit tail (the DamagingHit position, AFTER
      `apply_secondaries`). B2-1 `static_contact_proc_paralyzes_the_attacker` + B2-2
      `effect_spore_samples_a_status_onto_the_attacker` (the nested-sample draw pin).
    - **CONTACT recoil** (Rough Skin, `AbilityData.contact_recoil`) — DRAW-FREE `baseMaxhp/16` recoil to the
      attacker on a contact hit. B2-3 `rough_skin_recoils_the_attacker_draw_free` (STATE + the IDENTICAL-to-control
      seed proving draw-freeness).
    - **BLOCK** (`AbilityData.blocks_explosion`/`blocks_sound`/`blocks_phaze_drag`): **Damp** cancels Explosion /
      Self-Destruct at `runEvent('TryMove')` (battle-actions.ts:412, BEFORE the self-KO faint at 422 AND the
      accuracy roll) — the user does NOT self-KO, the move draws NOTHING (a big draw-count drop; `|cant|<damp
      holder>|ability: Damp|<Move>|[of] <user>`); fires for EITHER side's Explosion (`onAnyTryMove`), via
      `turn.rs::damp_holder`. **Soundproof** is IMMUNE to a SOUND move (`MoveData::is_sound`, the new `flags.sound`
      field — Sing / Grass Whistle in the status arm + Roar in the phaze arm): accuracy drawn, then `-immune|
      [from] ability: Soundproof`, no status / no drag / no sample (the same draw model as a type-immune move).
      **Suction Cups** blocks a phaze DRAG: the sim's `forceSwitch` runs `runEvent('DragOut')` in the MOVE BODY
      (battle-actions.ts:1166) → Suction Cups' `onDragOut` returns `null` → `forceSwitchFlag` NOT set → the
      runAction-tail `dragIn` never fires → **NO `sample` draw** (`-activate Suction Cups`, the holder STAYS), via
      the phaze arm's Suction-Cups gate. B2-4 `damp_cancels_explosion_no_self_ko` + B2-5 `soundproof_immune_to_sing`
      + B2-6 `suction_cups_blocks_the_roar_drag_no_sample`.
    - **SYNCHRONIZE** (`AbilityData.synchronize`) — when the holder is inflicted a MAJOR status by a FOE SOURCE (a
      status MOVE or a damaging move's SECONDARY), REFLECTS it back to that source (slp/frz EXEMPT; tox→psn). Wired
      at the SINGLE status choke point `turn.rs::try_set_status` (now `source: Option<(usize,usize)>`-threaded
      through its 3 real callers): after a foe-sourced major status applies to a Synchronize holder, it recurses
      `try_set_status(source, refl, None, …)` (source-less → no ping-pong). **DRAW-FREE in gen3customgame** (the
      reflected status draws no clause shuffle — probe-verified identical draws to a no-op control); in gen3ou it
      draws the reflected status's own 2-clause SetStatus shuffle. B2-7 `synchronize_reflects_paralysis_to_the_caster`.
    **DATA + move flags**: the CONTACT_PROC params + `contactRecoil`/`blocksSound`/`blocksExplosion`/
    `blocksPhazeDrag`/`synchronize` are extracted into `gen3_abilities.json` (`_GEN3_ABILITY_MECHANICS`, drift-gated
    by `dump_gen3_mechanics.js --check` which DERIVES the same from the resolved `onDamagingHit`/`onTryHit`/
    `onAnyTryMove`/`onDragOut`/`onAfterSetStatus` handlers), and the `contact` + `sound` move flags into
    `gen3_moves.json` (`flags.contact`/`flags.sound`). All obs-neutral (the Python `agents.gen3_data` facade ignores
    them; extractor-parity green). **e2e admission** grew the filter-clean pool **571 → 585 / 719** (`synchronize`
    [the #1 taxonomy gap] + `effectspore` the levers; STRICT `filtered_diverged == 0` over 220 battles / 11790
    decisions, a CLEAN pass first-try — **NO new engine bug**; the CONTACT_PROC draw-after-secondary + the BLOCK
    draw-count drops + the draw-free Synchronize reflect composed cleanly). Trace / Shed Skin / the berries
    have since shipped + been e2e-admitted in **batch 3** (`gen3_berry_trace_shedskin_v1`, 585 → 712 / 719 —
    see the roadmap item 3); Cute Charm + Color Change have since shipped in **batch 4** (below).
  - **BATCH-4 — the FINAL mechanics tail, DONE** (`gen3_ability_batch4_v1`, 2026-07-08). The last seven
    members, every draw model PROBE-settled (`harness/probe_{truant,truant_edges,innerfocus,shadowtag,
    cutecharm_attract,colorchange,kingsrock,kingsrock_order,focusband,focusband_confusion}_rng.js` +
    the shared `probe_batch4_lib.js`), validated by the class-sweep golden `gen_ability_batch4_golden.js`
    → `tests/ability_batch4_test.rs` (**21 scenarios × 60 seeds = 1260 game-end battles, 4220
    per-decision STATE+HP+STATUS+TRAPPED+SEED rows, byte-for-byte, md5-pinned**) + the **B4-1..B4-7**
    revert-verified pins in `regression_test.rs` (ground truth `harness/probe_batch4_regression_rng.js`):
    - **TRUANT** (`MonState::truant_turn`) — onBeforeMove priority **9** (slp/frz 10 > truant 9 >
      flinch 8): `|cant|…|ability: Truant` iff the flag, DRAW-FREE (a loaf turn draws NOTHING — no
      para roll [Q2b], no PP); `onSwitchIn` arms `turn !== 0`; the order-**27** residual TOGGLES it —
      so a mid-turn entrant (pivot/drag/action-faint replacement) is toggled back and MOVES its first
      full turn, while a POST-residual DoT-KO replacement keeps `true` and LOAFS (edge E1); a
      speed-tied Truant MIRROR adds ONE order-27 tie-shuffle draw (Q4's 9-vs-8).
    - **INNER FOCUS** — blocks the flinch volatile at the APPLY (the flinch-secondary `random(100)`
      STILL draws — draw-count-IDENTICAL to a landed flinch, probe-pinned vs a Thick Fat control;
      CONTRAST Shield Dust's filter-the-draw). One gate in `apply_one_secondary`'s flinch arm — it
      covers a move's own flinch AND the King's Rock appended one.
    - **SHADOW TAG** — `is_trapped` traps the foe UNCONDITIONALLY (no grounded/type gate — a Flying
      Skarmory is trapped; a MIRROR is MUTUALLY trapped — `onFoeTrapPokemon` has no fellow-holder
      exemption, only the display-only Maybe does), DRAW-FREE (a Wobbuffet mirror's draw count ==
      a no-trap control's; vs Magnet Pull's onAny* draws).
    - **CUTE CHARM + the ATTRACT volatile** (`AbilityData::contact_attract` +
      `MonState::{gender, attract}`) — the CC roll `randomChance(1,3)` draws UNCONDITIONALLY on a
      damaging contact hit (the GENDER gate lives INSIDE `attract.onStart` — F-into-F / genderless
      draw the roll, the volatile fails draw-free); attract: onBeforeMove priority **2** (confusion
      3 > attract 2 > par 1), `-activate` ALWAYS then `randomChance(1,2)` (cant on pass), NO
      duration, cleared when the SOURCE leaves the field (onUpdate) or the HOLDER switches out.
      Gender is parsed from the packed set; an UNSPECIFIED gender on a ratio species makes the SIM
      draw `battle.sample(['M','F'])` at construction (an unmodeled init draw) → the attract compare
      PANICS fail-loud on an unknown gender, and every golden pins genders explicitly.
    - **COLOR CHANGE** (`MonState::types_override` + the ONE **`mon_types`** choke point, which now
      serves EVERY live type read: STAB, chart effectiveness, status type-immunity, sand-chip
      immunity, Magnet Pull's Steel gate, Leech Seed's Grass gate) — onDamagingHit sets
      `[move.type]`: DRAW-FREE; NOT behind a sub (the mon's DamagingHit never fires — the batch-2
      lesson, probe-verified); not on the KO hit; never for typeless `???`; no-op on an
      already-matching type; switch-out reverts.
    - **KING'S ROCK + FOCUS BAND** — see roadmap item 4 (PROC_ITEM, WIRED).
    The batch surfaced NO new engine bug — the golden passed bit-for-bit first-try (two HARNESS
    fixes only: the first-mover scan now counts a voluntary `|switch|` as the first actor [matching
    the port's action queue], and one scenario was redesigned off the sim's accept-then-`|cant|nopp`
    0-PP path, which the port's strict request-legality gate deliberately rejects — scripted goldens
    stay within request-legal choices).
  - **SWITCH_OUT ability — DONE as a class** (`gen3_natural_cure_v1`, 2026-07-06). NATURAL CURE (the sole
    gen-3 member) cures an ALIVE outgoing holder's major status on switch-OUT (voluntary pivot OR
    phaze-DRAG-out; the tox stage + sleep counter reset), **DRAW-FREE** (`onSwitchOut`, `onCheckShow`
    undefined — resolving the long-deferred "NaturalCure CheckShow" draw question: NO CheckShow gate; the
    cure + its `[silent]` `-curestatus` reveal consume ZERO PRNG, so it is SEED-NEUTRAL; probe-settled by
    `harness/probe_naturalcure_rng.js` + `probe_naturalcure_dump.js`). It is an ENGINE FLAG (a `status =
    None` clear in `turn.rs::execute_switch`, gated on `naturalcure` + `!fainted`), NOT a `dmgMod` data row
    — `gen3_abilities.json` unchanged (obs-neutral). It was the **#1 e2e team-carry gap** (naturalcure=254);
    admitting it grew the filter-clean pool **151 → 449** — the biggest single lever yet — a CLEAN STRICT
    e2e pass (no new engine bug). Golden `gen_naturalcure_golden.js` → `naturalcure_test.rs` (280 game-end
    battles, the cure observable on the active-status timeline); pins `natural_cure_*` (NC1-NC3) in
    `regression_test.rs`; probes `probe_naturalcure_{dump,rng,scenario,regression_rng}.js`.
  - **TYPE-INTERACTION abilities — COMPLETE as a class** (Levitate immunity, Water/Volt Absorb
    heal+immunity, and now **Flash Fire's ×1.5 boost** — `gen3_flashfire_boost_v1`, 2026-07-06,
    the last gap). FF is NOT a `dmgMod` fold (its boost is a `flash_fire` activation VOLATILE +
    a ModifyDamagePhase1 damage fold, structurally like screens — an engine flag, not a data
    row), so it lives outside the DMG_MOD framework. The A/B fuzzer's evidence-based **#1 STATE
    cluster** (fireblast+flamethrower dominate the STATE repros, 397/402 with an FF mon); of 200
    replayed FF-team STATE repros, **185 (92.5%) flip to `ok`**. Golden `gen_flashfire_golden.js`
    → `flashfire_test.rs`; pins `flash_fire_*` in `regression_test.rs`; probe `probe_flashfire_rng.js`.


## The handler audit's first run — the two real misses

- **The first run surfaced TWO REAL MISSES, both fixed bit-for-bit + pinned:**
  1. **JUMP KICK / HIGH JUMP KICK crash** (`gen3_jump_kick_crash_v1`) — both passed
     `isModeledMove` (plain damaging moves) but carry an `onMoveFail`: a FAILED JK (miss or
     Protect block) crashes the USER for `clampIntRange(getDamage/2, 1, floor(TARGET.maxhp/2))`,
     and the crash's `getDamage` DRAWS crit + the 16-way roll (+2 draws vs a missed control).
     Probe-settled (`probe_jumpkick_crash_rng.js`): fires through Protect; NOT vs a
     Fighting-immune (Ghost) target; the crash can faint the user; Focus Band can survive it
     (a MOVE-effect Damage event). Fixed in `turn.rs::apply_jump_kick_crash` (the miss +
     protect-block returns); pins HA1 `jump_kick_miss_crashes_the_user_with_crit_and_roll_draws`
     + HA1b `jump_kick_crashes_through_a_protect_block` (revert-verified, sim ground truth
     `probe_handler_audit_regression_rng.js`). Zero e2e/A-B corpus exposure (no team carries it)
     — a pure latent bug.
  2. **FREEZE CLAUSE MOD** (`gen3_freeze_clause_v1`) — the engine modeled Sleep Clause but not
     Freeze Clause: under gen3ou a SECOND foe-inflicted freeze on a side must FAIL (the rule's
     `onSetStatus` returns false INSIDE the already-drawn SetStatus event — DRAW-FREE block;
     a fainted mon's status is `'fnt'`, so only LIVING frozen mons count). Probe-settled
     (`probe_freeze_clause_rng.js`); fixed in `turn.rs::try_set_status` + `side_has_frozen`
     (mirrors the sleep path, same `sleep_clause` format flag); pin HA2
     `freeze_clause_blocks_the_second_freeze_in_gen3ou` (revert-verified). Unreachable in the
     e2e/A-B (both run gen3customgame) — a latent bug for every clause format.
  Notable already-modeled rows the audit CONFIRMED (previously bug-class members): the
  fire-hit thaw (`frz.onDamagingHit`), Early Bird's double sleep decrement, Pressure's +1 PP,
  the Trace-route STATUS_IMMUNE onUpdate cures, Oblivious's attract gate, Rock Head's
  Struggle exemption (a true no-op — Struggle is the only modeled recoil).



---

## The extraction, the class map and the committed data

- **The extraction + the CLASS MAP.** `harness/dump_gen3_mechanics.js` reads the RESOLVED
  `Dex.mod('gen3')` (the WHOLE mod chain applied — the mod-chain law below), dumps every gen3
  item (132 = 128 gen≤3 + the 4 incense exceptions) + every gen3 ability (76, nums 1-76) with its
  resolved handler inventory + extracted parameters, classifies each (UNCLASSIFIED fails the
  dump), and writes **`tests/vectors/gen3_mechanics_inventory.md`** — the class map every future
  phase executes against (per-entry class / params / modeled-status / DRAW-vs-free tag).
  `--check` is the DRIFT GATE: it verifies the committed `data/pokemon/gen3_items.json` /
  `gen3_abilities.json` mechanics fields EXACTLY match the resolved dist (run it whenever either
  regenerates); `--json` emits the machine-readable extraction.
- **THE MOD-CHAIN LAW (the Light Ball cautionary tale).** gen3 resolves through gen4 → … → base,
  and later mods REPLACE and DELETE handlers: base Light Ball doubles Atk+SpA, the gen4 mod
  REWRITES it to an `onBasePower` double, the gen3 mod REWRITES it again to **SpA-ONLY ×2**.
  NEVER regex a single data file — extract from the resolved dist; the probe/golden against the
  real sim is the only oracle. (Same law as the taunt/disable durations.)
- **The data (the dex-style source of truth).** `tools/pokemon_data_extractor/sync.py` emits
  ADDITIVE, obs-neutral fields into `data/pokemon/gen3_items.json` (`typeBoost {type, mod:[num,
  den], fold: stat|basePower|basePowerDirect}`, `statMods {stat:[num,den]}` + `onlySpecies` +
  `untransformedOnly`, `choice`, `isBerry`) and `gen3_abilities.json` (`dmgMod {mod, fold, type/
  types, pinch, whenStatused, direct}` — DATA-ONLY until the ability class is wired) from a
  curated table derived via the dump (the callbacks are JS — invisible declaratively — so the
  table is curated like `_CURES_SELF_STATUS`, and the `--check` gate pins it to the dist). The
  4 gen4-named incenses are an explicit, documented exception to the extractor's gen filter (the
  sim applies them under gen3 formats; adding ENTRIES is obs-neutral — the obs encodes items by
  per-id `num` lookup, no enumeration index). `dex/items.rs` parses it all into `ItemData`.
