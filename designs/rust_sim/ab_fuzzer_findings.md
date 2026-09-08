# The A/B fuzzers — what each sweep found, and the fixes that closed it

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

The four continuous differential hunters (`ab_fuzz.js` omniscient state+seed, `--protocol` byte,
`bridge_ab_fuzz.js` per-side+request, `gen_sim_bridge_diff.js` external-consistency) each ran to a
green gate. This document holds their finding records: the fix queues, the emission-form sweeps,
the R- and T-numbered byte/seed bugs, the banked probe specs, the subsequence seed anchor, and the
`ourandom` generator's construction. The leaf keeps how to RUN each fuzzer, the allowlist clauses
that define when the green gate may pass, and the standing lessons.

---

## `ab_fuzz.js` — the fix queues (state+seed sweeps)

- **The first bounded smoke (2026-07-03, master seed 20260703 — REAL findings, the fix queue;
  NOT fixed inline; repros under `harness/ab_fuzz_out/smoke_*/divergences/`):**
  `pool` 100 battles → **100 ok / 0 diverged** (the port stays bit-for-bit on the e2e corpus
  under fresh seeds/choices). `randbats` 300 battles (21135 decisions, ~2500 battles/h) →
  267 ok / **27 diverged + 1 panic** (kinds seed=21 status=4 state=2 panic=1). `random`
  200 battles (11412 decisions, ~3300 battles/h) → 49 ok / **151 diverged** (kinds state=89
  seed=60 firstmover=2) — the modeled-predicate surface beyond the 22 real teams' movesets had
  simply never been exercised. Triaged clusters (suspected mechanism per repro decode):
  1. ✅ **FIXED (2026-07-09, `gen3_facade_v1`) — Facade ×2-when-statused + the runEvent-tail
     INTEGER-GUARD.** Facade carries the dist `onBasePower` (`chainModify(2)` when the user
     has a non-`slp` major status); `isModeledMove` never rejected `onBasePower`, so it was
     admitted but priced flat BP 70. **Probe-settled** (`harness/probe_facade_gen3.js`):
     psn/tox/par all ×2 (BP 140); brn ×2 AND the gen3 burn damage-halve STILL applies (gen3
     Facade does NOT ignore burn — max-roll 108 == the unstatused 108); a burned GUTS user
     composes Atk ×1.5 + halve-suppressed + BP ×2 (318); DRAW-FREE (4 draws both arms). The
     fix is a BP-CHAIN member pushed in `run_move` (id-gated per the fixed-damage precedent —
     `gen3_moves.json` has no onBasePower field). The probe ALSO overturned the old "a Direct
     item discards the BP chain" shortcut in `calc_damage`: Pink Bow (Normal ×1.1 DIRECT
     float) + poisoned Facade (Normal chain ×2) CO-FIRE, and `70 * 1.1 == 77` EXACTLY in f64
     → the sim's runEvent-tail guard (`relayVar === Math.abs(Math.floor(relayVar))`,
     battle.js:709) PASSES and the accumulated chain RE-APPLIES → BP 154, NOT 77. `damage.rs`
     now implements the EXACT integer-guard (a non-integer float like 75×1.1=82.5 still skips
     the chain). **Pinned** by the revert-verified
     `regression_test.rs::facade_status_doubles_bp_and_composes` (FA-a..FA-e incl. the FA-d
     bow-composition; ground truth `harness/probe_facade_defrost_regression_rng.js`); each of
     the two components revert-fails its pin. **Parity: 143/145 facade-team repros in
     auto_0709_0805 + 333/344 in auto_0708_0304 flip to `ok`.** NO admission change was
     needed — `isModeledMove` already admitted facade (and the committed e2e golden replays
     220/220 byte-identical, md5 unchanged).
  2. **Pink Bow / Polkadot Bow (+ the gen4-named incenses)** — `MODELED_ITEMS` lists `pinkbow`,
     `polkadotbow`, `oddincense`, `rockincense`, `roseincense`, `waveincense` as modeled ×1.1
     type-boosters, but the port's `resolve_atk_stat_mods` table implements NONE of them (repro:
     Polkadot-Bow Body Slam dealt ×1.1 in the sim, flat in the port — kind=state). FIX: add the
     bows (real gen2/3 Normal ×1.1); DECIDE the incenses (gen4 items the sim still applies under
     gen3customgame) — implement or drop from `MODELED_ITEMS`.
  3. **Accuracy/evasion STAGES are not folded into the to-hit roll** (~8 random-mode battles) —
     the predicate admits accuracy-drop secondaries (Mud-Slap's `boosts:{accuracy:-1}` is a
     "structured stat-boost" shape); the port tracks the stage in `boosts[5]` but `run_move`
     rolls `random_chance(accuracy, 100)` on the RAW move accuracy (repro: a double-Mud-Slapped
     Entei's Bite hit in the port but missed in the sim → kind=seed via the hit-path draws).
     FIX: apply the gen-3 acc/eva stage table in the to-hit computation (or reject acc/eva-boost
     secondaries from the predicate).
  4. **Substitute-turn seed cluster** (7 of the randbats seed divergences have a Substitute at
     the diverging decision — new interleavings on varied-level randbats teams the constructed
     sub golden + the 284 e2e sub decisions never hit).
  5. **Ice-move FREEZE status cluster** (4 randbats battles: the port freezes a mon the sim does
     NOT, with the seed still matching — an equal-count mis-ORDERED draw or a freeze-gating rule;
     e.g. Delibird Ice Beam into Chimecho on a Toxic turn).
  6. **Switch-boundary seed cluster** (5 randbats + 6 random battles diverge at a switch-only
     decision — a switching/entry draw-order case beyond the e2e's team pool).
  7. **The 1 panic** (randbats, Whirlwind battle): the port's phaze DRAG diverged from the sim's
     upstream, so a later recorded slot choice landed on the port-active's unmodeled **Wish** →
     the fail-loud panic (caught by the replayer's `catch_unwind`, reported as
     `verdict=panic` with the message — the loop survived). The panic is the SYMPTOM; the drag
     divergence is the bug.
  Plus ~2 `firstmover` divergences (wrong first mover, seed + state matching) — an ordering-
  without-draw case. The `random` mode's high rate is dominated by clusters 1–3; dedupe by
  mechanism, not by repro count.
- ✅ **THE RESIDUAL TAIL CLEARED (2026-07-10)** — clusters 4–7 (and everything else left in the
  gender-pinned corpus) are FIXED: re-triaging the complete `auto_0709_0805` (307 repros) with
  the current binary and root-causing every survivor found **SEVEN engine bugs**, after which
  the corpus replays **307/307 ok** (incl. the 4 fail-loud panics — their upstream drag
  divergences sat inside the fixed clusters). The bugs (all probe-settled, each with a
  revert-verified pin — full record in EDGE_CASES.md "✅ CLEARED — the A/B residual tail"):
  (1) `gen3_plus_minus_v1` Plus/Minus cross-field SpA ×1.5 (the gen3 `onModifySpA` scans
  `getAllActive()` — FOES included; the old NOOP classification's "partner-less in singles"
  never faced Plus against Minus; `plus`/`minus` are now MODELED, not no-op — the admission
  union is unchanged so the e2e golden is untouched); (2) `gen3_ff_wisp_absorb_v1` Will-O-Wisp
  into a non-Fire, status-free, un-subbed Flash Fire holder is ABSORBED (arms, no burn — incl.
  a TRACED FF); (3) `gen3_cloudnine_end_v1` Cloud Nine / Air Lock `onEnd` fires
  `eachEvent('WeatherChange')` at BOTH End sites — switchIn's alive-outgoing ability End
  (pre-swap) AND faintMessages' pre-`fainted=true` ability End — one tie-shuffle on a
  cached-speed tie (the dominant "icebeam tail" was really randbats Golduck-mirror boards);
  (4) `gen3_ff_frozen_no_absorb_v1` a FROZEN Flash Fire holder is NOT fire-immune (full draws,
  then the fire-move thaw); (5+6) `gen3_fnt_clears_status_v1` `checkFainted` sets
  `status="fnt"` AND `clearVolatile` zeroes the corpse's boosts — so the replacement
  instaswitch sort reads the PLAIN corpse speed (para/+6 erased → the mirror ties draw);
  (7) `gen3_statusimmune_onupdate_cure_v1` the 6 STATUS_IMMUNE members' `onUpdate` CURES the
  holder's matching status (reachable only via TRACE — a slept Porygon2 tracing Insomnia).
  Pins PM/FFW/CN1/CN2/FZ3/FN1/TC1 in `tests/regression_test.rs`; ground truth
  `harness/probe_{plus_minus_gen3,plusminus_ffwisp_regression_rng,residual_tail_regression_rng}.js`.
  Full suite 273/0 green; e2e md5 `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED.
- ✅ **THE STEADY-STATE TAIL CLEARED (fix-queue #4, 2026-07-10)** — the first all-fixes 12h run
  (`auto_0709_2205`) produced **9 divergences / 35,018 battles (0.026%, 0 panics)**; re-triage on
  the residual-tail binary: 4 already-fixed noise, 5 true survivors → **THREE engine bugs**, after
  which the corpus replays **9/9 ok** (auto_0709_0805 stays 307/307; auto_0708_1705 replays
  489 ok / 0 diverged — its 1130 panics stay the pre-gender-pinning attract fail-loud noise):
  (1) `gen3_faint_queue_order_v1` — `faintMessages` drains `faintQueue` in ENQUEUE order (each
  corpse fully processed before the next corpse's ability-End), so a mutual Explosion's
  self-KO'd USER is already inactive when the Cloud Nine target's End WeatherChange fires → the
  dying holder gathers alone, NO tie draw (the port's side-order walk drew a phantom shuffle);
  (2) `gen3_fainted_no_ability_speed_v1` — a corpse's ability handlers no longer gather: a Swift
  Swim corpse under rain sorts the replacement instaswitch at PLAIN speed (alive 368 → fainted
  184, tying the mirror corpse → the shuffle the port missed);
  (3) `gen3_tox_stage_persists_v1` — the gen3 tox stage-0 reset fires via the runSwitch-time
  `runEvent('SwitchIn')`, NOT the raw switch swap: it RESETS on any switch-in whose runSwitch
  RUNS, but PERSISTS when the queued runSwitch is CANCELLED by gen3 faint-cancels-all (a
  co-replacement's Spikes-faint — the ab_1166_22 Mew's lethal stage-2 chip). Bug 3 was ALSO the
  fix-queue-#3 Lens-2 lead (auto_0708_1705 rmrcqwc2c_ab_793_13, state@38 hp 81-vs-97 —
  revert-reproduced, fix-flipped: a real bug, not gender noise). Pins FQ1/FS1/TX1/TX2 in
  `tests/regression_test.rs` (all revert-verified; TX2 pins the reset's PLACEMENT); ground truth
  `harness/probe_fixqueue4_regression_rng.js` + `probe_tox_stage_switch.js`. NEW REUSABLE TOOL:
  `harness/probe_repro_simtrace.js` — replay ANY saved repro dir through the REAL sim with
  per-draw PRNG call-site instrumentation (`node harness/probe_repro_simtrace.js <repro-dir>
  [decFrom] [decTo]`) — the root-causing workhorse for this queue. Full suite **277/0** green;
  e2e md5 `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED. Full record: EDGE_CASES.md
  "✅ CLEARED — A/B fix-queue #4".

## `--protocol` — the byte bugs and the emission-form sweeps

**Byte bugs FIXED (revert-pinned in `tests/protocol_byte_fuzz_test.rs`, all observation-only — the full
seed suite stays BYTE-IDENTICAL, e2e md5 unchanged):**
- **BF1 typed-HP move-name leak** — the port rendered the typed dex name `Hidden Power Ice`; gen-3 HIDES
  the HP type → `run_move` canonicalizes any `hiddenpower*` id to the bare `Hidden Power` for the announce.
- **BF2 Toxic residual `[from]` cause** — the DoT chip was `[from] tox`; Showdown reports `[from] psn`
  (the HP-field status token stays `tox`). A LATENT gap (the constructed protocol golden never realized a
  Toxic residual).
- **BF3 self/side-move announce target** — a NON-foe-directed move (`allySide`/`all`/`allyTeam` — Light
  Screen / Sunny Day / Rain Dance / Perish Song / Heal Bell) renders the USER as the `|move|` target, not
  the foe (`status_move_announce_renders_user`).
- **BF4 Pursuit interrupt `[from] Pursuit`** — the interrupt strike's `|move|` announce now folds
  `|[from] Pursuit` (via `set_next_move_from`).

**The STATUS-MOVE EMISSION-FORM SWEEP (2026-07-17) — pool byte-clean rate 27% → ~95%.** A pool-mode
`--protocol` fuzz over both formats surfaced ~8 general status-move / did-nothing emission forms that hit
nearly every real-team battle. All FIXED + revert-pinned in `protocol_byte_fuzz_test.rs` (13 pins;
`sleep_move_status_carries_from_move` / `toxic_into_steel_reports_immune` / `natural_cure_emits_curestatus_on_switch_out`
/ `recover_at_full_hp_emits_still_and_fail_heal` / `beat_up_emits_per_strike_activate_in_customgame` /
`protect_blocks_status_move_before_the_immune_report` / `shiny_mon_shows_the_shiny_details_flag` /
`knock_off_hint_fires_once_per_battle` / `stat_drop_blocked_by_substitute_emits_still_and_fail`), **all
observation-only — the full seed suite stays BYTE-IDENTICAL (cargo test green, e2e md5 unchanged):**
- **`||[still]` did-nothing FAIL framing + the `-fail` line** (the sim `attrLastMove('[still]')`s the
  announce + `runMoveEffects` `-fail`): Recover/Soft-Boiled@full → `[still]` + `-fail|<user>|heal` (the
  `heal` sub-tag); Toxic-vs-Substitute / **STAT-DROP-vs-Substitute** (Screech/Charm/Metal Sound/&c. into a
  non-`bypasssub` sub — the FORM-1 residual the byte fuzzer surfaced last, `BF-F15`, captured
  `harness/probe_statdrop_substitute.js`; the stat-drop arm's sub-block used to return emitting NOTHING) /
  repeat-Protect-or-willAct-fail / double-Wish / no-bench Roar /
  Refresh-no-status / weather-set-into-same (Rain Dance) / Light-Screen-already-up / Beat-Up-fizzle →
  `[still]` + BARE `-fail|<user>`. (The Protect willAct-fail — a Protect after the foe SWITCHED — also
  emits `-fail`; the double-Wish `-fail` corrected a batch-3 comment that wrongly said "no -fail".)
- **status-TYPE-immunity `-immune`** (Toxic→Steel/Poison, Will-O-Wisp→Fire) — the `try_set_status`
  `status_type_immune` gate emits `|-immune|<target>` when the source is a status MOVE (`announce_immune_block`
  == the sim's `sourceEffect?.status`); a secondary-inflicted type-immune status stays silent.
- **Protect-before-immunity ORDER** — `run_status_move` reordered to match gen3 `tryMoveHit`: on a HIT the
  `runEvent('TryHit')` handlers (Protect → `-activate Protect`; Soundproof) win BEFORE the naturalImmunity
  `-immune`; on a MISS naturalImmunity still wins (`-immune`, no `-miss`). So Thunder Wave into a
  Ground-typed PROTECTING foe shows `-activate Protect`, not `-immune`.
- **sleep `-status …|[from] move: <Move>`** — threaded the source move name into `try_set_status_impl`; only
  SLEEP carries it (par/psn/brn/frz from a move stay bare, per `conditions.js` per-status `onStart`).
- **Natural Cure `-curestatus …|[from] ability: Natural Cure|[silent]`** — emitted in `execute_switch`
  BEFORE the replacement `|switch|`/`|drag|` line (`curestatus_from_ability_silent`).
- **confusion `-start|X|confusion` / `-activate|X|confusion` / `-end|X|confusion` + the self-hit
  `-damage|…|[from] confusion`** — the `add_confusion` onStart, the `on_before_move` reveal + the counter-0
  `onEnd`, and the self-hit damage line.
- **fire-move thaw `-curestatus|<t>|frz|[msg]`** (`frz.onDamagingHit` `cureStatus()`).
- **the delta-0 `-boost`/`-unboost` at the ±6 cap** — a PRIMARY self-boost MOVE emits `|-boost|…|spe|0`
  (Agility@+6) via `boost_applied` (the sim's `boost()` `!isSecondary && !isSelf` branch); the Clear
  Body / White Smoke / Hyper Cutter `-fail|unboost|[from] ability|[of]` for a PRIMARY foe-drop (Screech).
- **CONTACT-PROC status carries `[from] ability: <A>|[of] <holder>`** (Static/Poison Point/Flame Body/Effect
  Spore) — a CORRECTED earlier claim (this note used to say the form was BARE; the live sim capture refuted
  it, `gen3_omniscient_byte_fuzz_v1`, golden `|-status|p1a: Metagross|par|[from] ability: Effect Spore|[of]
  p2a: Breloom`). The sim's `source.trySetStatus(status, target)` runs INSIDE the ability's `onDamagingHit`,
  so `setStatus`'s null `sourceEffect` falls back to `this.battle.effect` = the ABILITY → the status
  `onStart`'s `sourceEffect.effectType === 'Ability'` gate is TRUE → the `[from] ability` form. `apply_contact_proc`
  now threads the ability name through `try_set_status_impl`'s `ability_reveal` (the generalized Synchronize
  path). Draws unchanged (emission-only); all seed goldens byte-identical.
- **Volt/Water Absorb `-heal` (not `-immune`) when the heal LANDS** — `apply_absorb_heal` now reports
  whether it healed; a below-full holder shows `-heal|…|[from] ability: Volt Absorb|[of] <user>`, only a
  full-HP holder shows `-immune|…|[from] ability` (the F3 capture had only exercised the full-HP case).
- **Beat Up per-strike `-activate|<user>|move: Beat Up|[of] <ally>`** — gen3customgame EMITS it (the mod's
  `condition.onModifySpA`, gated on `!beatupnicknamesmod` — a gen3 Standard rule present in gen3ou, absent
  in customgame → aligned with `sleep_clause`). The task's "gen3 does not emit it" claim was WRONG; the
  omniscient stream + resolved dist confirm the customgame emit.
- **shiny details flag** (`|switch|…|Quagsire, M, shiny|…`) — `switch_details` appends `, shiny` from
  `set.shiny`.
- **`-hint` dedup mirrors the sim's `once` param (`ProtocolBuilder::hint(text, once)`)** — the sim's
  `Battle.hint(hint, once, side)` (battle.ts:3092) returns early if the text is already in `this.hints`,
  but only ADDS the text `if (once)`. So a `once: true` hint fires ONCE per battle (**Knock Off**,
  gen4/moves.ts:703), while a `once: false` hint fires on EVERY occurrence (**Sleep Clause Mod**,
  rulesets.ts:1395; the gen3 **Intimidate-vs-Substitute** + **Pursuit** notes, no `once` arg). The port
  used to dedup ALL hints (a `HashSet` insert unconditionally) → a gen3ou **double sleep-clause block**
  emitted the hint ONCE where the sim emits it twice (the byte-fuzzer-surfaced 4th residual). FIXED —
  `hint` now takes `once: bool` (dedup only when `once`); the four call sites pass the sim's value.
  Observation-only → all seed goldens byte-identical.
- **gen3ou clause `-message`** — a Sleep-Clause-blocked 2nd sleep emits `|-message|Sleep Clause Mod
  activated.` + the `once:false` `|-hint|` (once PER block, not deduped); a Freeze-Clause-blocked 2nd
  freeze emits `|-message|Freeze Clause activated.` (rulesets.js). gen3customgame has no clauses → no
  message (so the e2e/seed suites, all gen3customgame, are untouched).

**The WIDE-NET SWEEP round (2026-07-17, round 8) — 7 more emission forms + a bridge-request allowlist gap.**
A broader fuzz (`--mode random` = every gen3 species/ability + `pool` both formats + new master-seeds)
flushed rare forms the bounded pool missed. All SIM-PROBED (the sim is the oracle), FIXED, and pinned by
revert-verified CONSTRUCTED-scenario tests in `protocol_byte_fuzz_test.rs` (BF-F16..BF-F20 + the shared
builder pins); ALL observation-only — the full seed suite stays BYTE-IDENTICAL (cargo test 61/61 green,
e2e md5 unchanged):
- **BF-F16 the two-turn charge `[from] lockedmove` SPACE** — a locked Solar Beam fire-turn announce carries
  `|move|…|[from] lockedmove` WITH a space (the sim's `attrLastMove('[from] lockedmove')`); the port emitted
  the no-space `|[from]lockedmove`. Hits the real gen3ou POOL (Houndoom/Shiftry SolarBeam) at BOTH formats.
  ALSO the fire-turn-MISS ORDER: the sim appends `[from] lockedmove` (useMove) BEFORE `[miss]` (accuracy) →
  `|…|[from] lockedmove|[miss]`; the miss branch now emits the bare move line → lockedmove attr → miss attr.
- **BF-F17 SPEED BOOST `-ability` announce** — the end-of-turn Speed Boost residual emits
  `|-ability|<mon>|Speed Boost|boost` (the sim's ability-source `boost()` announce) BEFORE the
  `|-boost|<mon>|spe|1`; the port dropped it. Emitted only when the stage actually rose (a +6-cap draws
  nothing, both lines).
- **BF-F18 HYPER CUTTER / KEEN EYE `-fail` STAT token** — a SINGLE-STAT boost-blocker carries its stat
  token: Hyper Cutter → `unboost|Attack`, Keen Eye → `unboost|accuracy` (`unboost_fail_stat_token`); the
  whole-table blockers (Clear Body / White Smoke) carry none. The port dropped Hyper Cutter's `Attack` →
  a divergence on any Intimidate/Charm/Feather-Dance-into-Hyper-Cutter.
- **BF-F19 COLOR CHANGE typechange CASE** — `|-start|<mon>|typechange|<Type>|…` renders the DISPLAY-cased
  type (`Type::display_name`, e.g. `Psychic`), NOT the internal UPPERCASE key (`PSYCHIC`).
- **BF-F20 the confusion-berry DOUBLE `-start|confusion`** — a Figy/Mago/Iapapa/Aguav/Wiki confusion berry
  emitted its `-start confusion` TWICE (once inside `add_confusion`, once at the berry site); the redundant
  berry-site emission is removed → EXACTLY one.
- **the EFFECT-SPORE SLEEP over-attribution** — the gen3 `slp.onStart` (`mods/gen3/conditions.js`) DROPS
  the base `[from] ability` branch (only a MOVE source carries `[from] move: <Name>`), so Effect Spore
  inflicting sleep emits a BARE `|-status|<mon>|slp`; the port over-attributed it `[from] ability: Effect
  Spore|[of]`. (par/psn/brn/frz from an ability STILL carry `[from] ability` — the port already matched.)
  **PINNED (round-8 FIX)** by the revert-verified
  `protocol_byte_fuzz_test.rs::effect_spore_sleep_status_is_bare_not_ability_attributed` (Muk Tackles an
  Effect-Spore Breloom, seed `40,96,171,230` forces a proc that samples slp).
- **FUTURE SIGHT fainted-target `-hint`** — a future move resolving against a FAINTED slot occupant emits
  `|-hint|<Move> did not hit because the target is fainted.` (`futuremove.onEnd`, `once` falsy → no dedup);
  the port skipped it (formerly "uncaptured"). **PINNED (round-8 FIX)** by the revert-verified
  `protocol_byte_fuzz_test.rs::future_sight_resolving_against_a_fainted_slot_emits_the_hint` (Tyranitar
  Sand Stream casts Future Sight, chips Growlithe, the RESOLVE-turn sand chip [order 8] KOs it before the
  order-11 resolve → fainted slot → the hint; seed `19,47,80,111`; sim text confirmed vs `conditions.ts:399`).
- **the HEAL BELL / AROMATHERAPY bench-cure IDENT (bonus, random-mode find)** — a Heal Bell curing a
  STATUSED BENCH ally emits `|-curestatus|pN: <mon-name>|<status>|[silent]` (the mon's nickname/species,
  position-less), NOT the player-name `pN: <PlayerName>` the port rendered via `side_ref`.
- **[8]/[9] the bridge `|request|` allowlist gap** — the `return102`/`frustration102` numeric-BP alias and
  the non-Ghost Curse `target:self` ESCAPED the per-side gate on a CO-OCCURRING Curse+Return team: the sim
  renders the alias INCONSISTENTLY (roster `return102`, active id BARE `return`, active display `Return 102`
  with a SPACE — SIM-PROBED), so the old single-direction `replace("return","return102")` OVER-corrected the
  active id AND missed the display, leaving a residual that made the WHOLE reconcile (incl. the correct
  Curse-target fix) return None → both escaped. FIX (`bridge_replay.rs::classify_known_perside_residual`,
  gate-only): NORMALIZE BOTH sides by collapsing every alias form to the bare token before comparing (poke-env
  resolves both). Pinned by `perside_request_residual_tests` (co-occurrence allowlisted + a genuine-diff
  NOT swallowed). Verified GREEN: omniscient pool both formats (0 non-allowlisted) + `random` mode confirms
  the 7 emission forms GONE + bridge pool 0 diverged (curse-target 73 / return102 10 correctly allowlisted).
  ~~HONESTLY-OPEN random-mode-only residuals (NOT gen3ou POOL, deferred): a phaze `[miss]` on an
  evasion-miss, and the Damp-cancels-Self-Destruct `[still]` form.~~ (The Protect-vs-Soundproof TryHit
  order was FIXED in round-8 FIX below.) ⚠️ **BOTH of those are now CLOSED and this line is HISTORICAL** —
  the Damp `[still]` announce in **ROUND 22** (fixture `42_damp_blocks_explosion_move_still.txt`), the
  phaze/evasion + Attract `-end` forms across **ROUNDS 25/26**. Round 30 re-measured the benchmark and
  found **0 protocol divergences**; do NOT re-open these from this paragraph without re-measuring first.


## The R- and T-numbered byte/seed bugs behind the green gate

**R3 (IV-derived Hidden Power BP) — FIXED in the engine (`gen3_iv_derived_hidden_power_bp_v1`), quarantine
REMOVED.** gen-3 computes Hidden Power's base power from the ATTACKER's IVs (`Dex.getHiddenPower`,
`⌊hpPowerX·40/63⌋+30`, range 30..=70), NOT the flat 70 the data ships (all 16 typed HP rows are BP 70 in
`gen3_moves.json`). The port used to read the data's 70, over-damaging any real gen3ou HP mon whose IVs give
BP != 70 (a -1 Atk-IV spread → BP 68 → ~2/68 ≈ 3% over-damage that cascaded KO thresholds — the ~1.5%-of-pool
byte-fuzz STATE divergence). FIX: `state.rs::hidden_power_bp(ivs)` precomputes each mon's `MonState.hidden_power_bp`
in `from_set` (the WEIGHT-ORDER CRUX: the sim iterates `{hp,atk,def,SPE,SPA,spd}` so Speed carries weight 8 —
BEFORE SpA 16 / SpD 32 — while the port's IV array is `[hp,atk,def,spa,spd,spe]`; a naive array-order mapping is
WRONG; GIGO-guarded to 30..=70). `run_move` overrides the data BP with it for any `hiddenpower*` move (the TYPE
stays from the typed id); the bridge request-JSON move name reads it too (the sim renders `Hidden Power <Type>
<IV-power>`). It is a bridge-correctness fix only — the Python RL obs / DamageOperator keep their own
internally-consistent BP-70 assumption (untouched). Confirmed bit-for-bit vs `getHiddenPower` (source read) AND a
live damage probe. The `ab_fuzz.js` pool `teamHpBp70Clean` picker quarantine is REMOVED — every real gen3ou HP
team is now byte-safe. STATE-fix but OBSERVATION-NEUTRAL for the committed goldens (the e2e never PICKS HP —
`isModeledMove` defaults `allowHiddenPower=false` — and the constructed goldens use IV-31 → BP 70 → unchanged; the
e2e golden md5 `3155eb796cb4bf453c6053d769ba98e5` is UNCHANGED). Pins: `hidden_power_bp_is_iv_derived_not_flat_seventy`
(a constructed BP-68 HP Ice → the sim's 53 dmg, revert-verified) + `hidden_power_bp_weight_order_and_boundaries`
(the SPE-8-not-32 crux + 30..=70 boundaries); ground truth `harness/probe_hidden_power_bp_regression_rng.js`.

**R2 (the Leftovers `-heal` emit ORDER on a residual speed TIE) — FIXED (`gen3_leftovers_slotcond_gather_order_v1`).**
Root-caused via `harness/probe_r2_repro_trace.js` on a byte-fuzz repro (a Jolteon mirror, both Leftovers,
sandstorm, a pending p2 Wish): `speed_sort` is a NON-STABLE selection sort whose swaps DISTURB the relative order
of the tied handlers, so the tie-group Fisher-Yates shuffle reads whatever pre-sort order the swaps LEFT the tied
pair in. Showdown gathers a side's SLOT CONDITIONS (Wish order 7 / Future Move order 11) via
`findSideEventHandlers(…, active)` — AFTER that active's pokemon handlers — so Wish sits AFTER Leftovers in the
pre-sort array. The port gathered Wish/FutureMove FIRST (a pre-loop at the array front), so the selection-sort's
Wish/weather swaps REVERSED the tied Leftovers pair vs the sim → the two `-heal` lines emitted in the OPPOSITE
order at the SAME shuffle value (the seed matched — the Sodium seed depends on the draw COUNT, not the
permutation). FIX: `run_residuals` now gathers the Wish + FutureMove handlers PER-ACTIVE at the end of the
per-active loop (after the item), mirroring the sim. DRAW-NEUTRAL (same handlers / sort keys / tie-group COUNT →
the seed is unchanged; only the emit permutation moves to match Showdown) → OBSERVATION-ONLY, every seed +
protocol + writeline golden stays BYTE-IDENTICAL. Pin: `leftovers_heal_order_follows_the_slot_condition_gather`
(a Jolteon-mirror + pending-Wish scenario whose `-heal` marker sequence == Showdown's, revert-verified) + the
byte-fuzz corpus fixture `23_leftovers_wish_heal_order.txt` (the repro replays byte-clean; reverting the fix →
`kind=protocol` at the exact `-heal` line, fault-injection proven); ground truth
`harness/probe_r2_wish_leftovers_regression_rng.js`.

**R13 (the ENCORE × SLEEP TALK draw-count desync) — FIXED (`gen3_encore_sleeptalk_trylhit_v1`, the pool
byte-fuzz `--protocol` gen3ou find ab_15_15 @ master-seed 222333, dec 47 — `kind=seed`).** Root-caused via
the FIXED `harness/probe_repro_simtrace.js` (in `--format gen3ou` — the default is customgame, so a
gen3ou repro MUST be replayed with `{format:'gen3ou', allowHiddenPower:true}` or the sim diverges from the
golden) + an env-gated per-draw PRNG backtrace in the port: at dec 47 the port drew **7** where the sim
drew **3** (+4). Suicune (asleep, lastMove Sleep Talk) is Encored by a FASTER Jumpluff THIS turn (Encore
locks Sleep Talk); the port's Encore `onOverrideAction` redirects Suicune's queued Rest → the encored Sleep
Talk slot, then RAN Sleep Talk (sampled a move + ran it, +4 draws). The SIM instead FAILS Sleep Talk
DRAW-FREE: Sleep Talk's resolved **`onTryHit(pokemon){ return !volatiles['choicelock'] && !volatiles['encore']; }`**
returns false when the user carries the `encore` volatile → `|move|…Sleep Talk||[still]` + `|-fail|`, NO
sample. (probe-confirmed: the sim's `singleEvent('TryHit', sleeptalk)` → false at turn 43, activeMove
sleeptalk, 0 draws; the golden protocol shows `cant slp` + `Sleep Talk [still]` + `-fail`.) So an
Encored-into-Sleep-Talk mon can NEVER resolve Sleep Talk while the encore holds. The port's sleeptalk arm
gated ONLY on `choicelock` (via `was_choice_locked`) — batch 5 wrote "Encore is unmodeled gen-3-wide" as
the reason, but **batch 6 later MODELED Encore and the gate was never updated** — the latent bug the pool
fuzzer surfaced. FIX (`turn.rs`, the sleeptalk arm): `if was_choice_locked || encore.is_some()` fails Sleep
Talk draw-free (the encore is read at its CURRENT value, NOT a pre-move snapshot, because the foe's Encore
can land THIS turn before the sleeper's Sleep Talk — matching the sim's live `volatiles['encore']` read).
NOT a clause-path fix, but OBSERVATION-NEUTRAL for every committed golden (no gen3customgame e2e/seed board
pairs an Encored mon USING Sleep Talk → full suite byte-identical, **e2e md5 `3155eb796cb4bf453c6053d769ba98e5`
UNCHANGED**, every seed golden byte-identical). Pinned by the revert-verified
`regression_test.rs::encored_mon_sleep_talk_fails_draw_free_via_ontryhit` (a fast Electrode Encores an
asleep Snorlax RestTalker → the encored Sleep Talk fails draw-free; ground truth
`harness/probe_r13_encore_sleeptalk_rng.js`, raw seed [7,11,13,17]). RE-VERIFIED: the 222333 pool
`--protocol` gate is now GREEN both formats (gen3ou 0 non-allowlisted, ab_15_15 → `ok`; gen3customgame
unchanged).

**R15 (the SLEEP-CLAUSE × self-REST-sleeper draw-count desync) — FIXED
(`gen3_sleep_clause_self_rest_exempt_v1`, the pool byte-fuzz `--protocol` gen3ou find ab_3_15 @
master-seed 333444, dec 24 — `kind=seed`, expected `56120,…` got `20330,…`; DISTINCT from R13).**
Root-caused via the R9-fixed `probe_repro_simtrace.js` + a temporary env-gated per-draw backtrace in
`prng::next` (removed): the port did **one EXTRA draw** at dec 24 (a second SetStatus clause shuffle).
The dec-24 turn is p1 Breloom **Spore → p2 Suicune** while p2's benched Zapdos is asleep FROM ITS OWN
turn-1 **Rest**. gen3 **Sleep Clause Mod** (`rulesets.ts` `sleepclausemod.onSetStatus`) counts an
existing sleeper toward the one-foe-asleep cap ONLY when `!pokemon.statusState.source?.isAlly(pokemon)`
— a mon that put ITSELF to sleep via Rest is EXEMPT (probe-verified `harness/probe_r15_sleepclause.js`:
a self-Rested Zapdos does NOT block a foe's Spore on Suicune → Suicune gets `slp`). WRONG (pre-fix):
`turn.rs::side_has_sleeper` counted ANY asleep mon (a stale "Rest is out of scope" assumption from
before Rest was modeled), so the port clause-BLOCKED the Spore; Suicune's own Rest then ran, drawing an
EXTRA clause shuffle (+1 draw → the seed divergence + a silent Suicune-state divergence ab_replay's
active-only state check missed). The fix: a new `MonState::sleep_from_rest` (appended LAST; set `true`
in `run_rest`, `false` in `try_set_status_impl` — mirrors `statusState.source.isAlly`), and
`side_has_sleeper` skips self-Rest sleepers. Revert-pinned:
`regression_test.rs::sleep_clause_exempts_a_self_rest_sleeper_from_blocking_a_foe_sleep_move`
(gen3ou, the sim's first-decision seed 53118,…; ground truth `harness/probe_r15_pin_truth.js` — post-T2
seed `21514,3448,20660,22314`, FAILS on revert). SEED-NEUTRAL for every committed golden: e2e md5
UNCHANGED `3155eb796cb4bf453c6053d769ba98e5`, all seed suites byte-identical (id-gated: only a gen3ou
board with a benched self-Rest sleeper AND a foe sleep move on a different mon hits it). The 333444 pool
`--protocol` gate is GREEN both formats (gen3ou ab_3_15 → `ok`, diverged 0; gen3customgame diverged 0).

**T1 (the FREEZE-PERSISTENCE deep-STATE bug) — FIXED (`gen3_omniscient_byte_fuzz_v1`, byte-fuzz repro
rmroh04is_ab_4_18 / _4_8 + cg rmrohcsti_ab_4_18/_11_23).** The round-1 "deep state/HP divergence" (the port at
FULL HP where the sim was not) root-caused to the port OVER-THAWING a frozen mon hit by **Hidden Power Fire**
(or Weather Ball). gen3 `frz.onDamagingHit` (conditions.ts:45-50) thaws ONLY when
`this.dex.moves.get(move.id).type === 'Fire'` — the BASE-dex move type — with the explicit "don't count Hidden
Power or Weather Ball as Fire-type" comment (`dex.moves.get('hiddenpower').type === 'Normal'`, `'weatherball'
=== 'Normal'`). The port computed `is_fire` from the RESOLVED runtime type (Fire for the typed-HP nums
355-370), so HP Fire WRONGLY thawed. It surfaced as a kind=status "sim=Freeze / port=None" (ab_replay compares
only the ACTIVE mon's status per decision, so the lost freeze is caught only when the frozen mon becomes active
again). FIX (turn.rs, the `is_fire` decl): `move_type == Some(Type::Fire) && category != Status &&
!to_id(&m.id).starts_with("hiddenpower") && to_id(&m.id) != "weatherball"`. `is_fire` is used ONLY at the thaw
(grep-confirmed decl + the thaw site) — Flash Fire absorb / fm_frozen read `move_type == Some(Type::Fire)`
DIRECTLY (the resolved type), so a Flash Fire holder STILL correctly absorbs HP Fire. STATE fix but
OBSERVATION-NEUTRAL for every committed golden (no committed golden pairs a frozen mon with an HP-Fire/Weather-
Ball hit → full suite byte-identical, e2e md5 `3155eb796cb4bf453c6053d769ba98e5` UNCHANGED). Pins (revert-verified):
`hidden_power_fire_does_not_thaw_a_frozen_defender` (STATE: still frozen — revert → thaws → None) +
`flamethrower_does_thaw_a_frozen_defender` (control: a base-type-Fire move DOES thaw) +
`flash_fire_still_absorbs_hidden_power_fire` (control: the narrowing didn't break the FF absorb); corpus fixture
`26_freeze_persists_vs_hp_fire.txt`. The 4 freeze repros (4_18/4_8/11_23 + the freeze-adjacent seed 10_18) flip
to ok.

**Two EMISSION-FORM byte fixes (observation-only, seed suite BYTE-IDENTICAL) — the actual round-2 protocol
tail (the round-1 T2-T6 list did NOT reproduce in the 600-battle pool run; do not trust it):**
- **ENDURE survive-at-1 `-damage` at exactly 1 HP** (repro rmroh04is_ab_8_4 / cg): when an endurer is ALREADY at
  1 HP, the endure clamp nets 0 damage (the mon stays at 1), so the caller's `realized > 0` emission gate SKIPPED
  the `|-damage|<mon>|1/<max>` line the sim STILL emits after `|-activate|move: Endure`. FIX: `endure_clamp`
  emits the `-damage` itself when it clamps to 0 (apply_damage(0) is a no-op → the current HP == post-apply HP),
  covering all 4 endure sites; the `hp > 1` path stays caller-emitted (no double). Corpus fixture
  `24_endure_survive_at_one_hp.txt`.
- **NATURAL CURE `-curestatus` on a Pursuit-KO'd switcher** (repro rmroh04is_ab_10_1): the gen 2-4 Pursuit
  interrupt runs the switcher's SwitchOut event (hence `naturalcure.onSwitchOut`) on the 0-HP-but-NOT-YET-fainted
  mon, so its tox/etc is CURED with a `|-curestatus|<mon>|<tok>|[from] ability: Natural Cure|[silent]` line BEFORE
  the `|-hint|`/`|faint|`. The port's Pursuit path called `process_faints` (setting `fainted`) BEFORE the switch,
  so the later clearVolatile Natural-Cure block (`!fainted`-gated) skipped it. FIX: run the NC cure on the
  hp==0-not-fainted switcher in the Pursuit-interrupt block, BEFORE the hint + process_faints (state-neutral —
  the mon faints anyway). Corpus fixture `25_natural_cure_pursuit_ko_curestatus.txt`.

**HONESTLY-OPEN residual TAIL (round-2, POST the T1 + emission + allowlist-narrowing fixes; master-seed 424242,
ISOLATED build).** gen3ou 300 battles: **7 → 4** non-allowlisted; gen3cg 300: **5 → 3**. The remaining are TWO
classes:
- **2 REAL seed (draw-count) bugs — both ✅ FIXED (round-4, `gen3_pressure_allyteam_v1` PA3):** the ROOT of
  BOTH **5_6** (both formats, ou dec159 + cg dec143) AND **3_24** (gen3ou, ou dec147) was the SAME
  **Pressure-over-deducts-PP-on-a-non-Ghost-Curse** bug — NOT the "Roar-loop PHAZE" / "Magnet-Pull switch"
  labels (those were wrong guesses). gen-3 `curse.onModifyMove` RE-TARGETS a NON-Ghost user's Curse to `self`
  (`nonGhostTarget`), but the STATIC dex `target` is `"normal"`, so `pressure_targets_foe` read `"normal"` and
  deducted 2 PP under a Pressure foe instead of 1 → the Curse-slot PP drained ~1 cycle early → forced Struggle
  turns the sim still Curses → the deep seed desync + cascade (the recorded protocol-diff was `|move|…|Struggle`
  vs `|move|…|Curse`; Curse counts SIM 16× / PORT 10× + 6 Struggles). FIX: `turn.rs`'s move-resolution tuple
  now computes the RUNTIME-effective Curse target (`is_nonghost_curse` → `"self"`) and feeds it to BOTH
  `targets_self` and `pressure_targets_foe`, so a non-Ghost Curse deducts 1 PP under Pressure. A NEW third case
  of the `gen3_pressure_allyteam_v1`/`gen3_pressure_foeside_v1` class. Pin: revert-verified
  `regression_test.rs::pressure_does_not_add_pp_for_a_nonghost_curse` (Zapdos(Pressure)-vs-Swampert(Curse):
  Curse PP 16→15 not →14 + the dec7 post-decision seed; ground truth
  `harness/probe_pressure_curse_regression_rng.js`). Verified: the 424242 pool run (both formats) now leaves
  ONLY the 2 harmless construction `[of]`-flip artifacts (5_6 / 3_24 / the cg dec-30 Tyranitar-sandstorm
  cascade all GONE — the cascade shared this same Pressure-Curse root).
- **The construction weather-`[of]`-flip protocol artifacts (2/format, harmless) — ✅ now NARROWLY ALLOWLISTED
  (round-6, A1 `turn0-construction-speed-tie-mirror-of-flip`, see the A1 entry above).** 4_8 / 11_20 (both
  formats) are single-line `-weather|…|[of] p1a` vs `[of] p2a` mirror flips (seed=None-invisible, zero obs
  impact) — the follow-on narrow signature the round-2 note anticipated. The 6-clause key allowlists ONLY this
  exact same-species-mirror `[of]`-flip form; injection-proven not to swallow a wrong-`[of]`-to-a-real-different-
  mon. So the OMNISCIENT gate is now GREEN both formats (0 non-allowlisted).

The round-1 T2-T6 list (`-miss` p2:/p2a:, Light Screen `-sideend`/`-resisted` order, faint-vs-`|upkeep|` order,
Solar Beam `|[from] lockedmove` gap, Morning Sun `||[still]`) did NOT reproduce as a non-allowlisted divergence
in this 600-battle pool run — either already resolved by the intervening emission sweep or lower-frequency;
NOT blind-fixed.


## `bridge_ab_fuzz.js` — the three Phase-1 bugs it found

- **Three real Phase-1 bugs it found + FIXED** (all probe-settled vs the sim, revert-verified):
  1. **`gen3_shadowtag_firm_trap_v1`** — the gen3 mod's **Shadow Tag** sets `pokemon.trapped = true`
     DIRECTLY (`onFoeTrapPokemon`), so its FIRST `move` request already carries `trapped:true` (NO
     `maybeTrapped` phase), and a rejected switch draws `|error|[Invalid choice]…` with NO re-request
     (the `emitChoiceError` update no-ops — nothing to firm). Arena Trap / Magnet Pull call
     `tryTrap(true)` → `trapped = 'hidden'` → the `maybeTrapped`→`[Unavailable choice]`+re-request
     machine. `state::trap_is_firm` distinguishes them; `bridge.rs::serialize_active` + the rejection
     round read it. Pinned by the `shadow_tag_firm_trap` scenario in `bridge_trapping_golden.txt`
     (`gen_bridge_trapping_capture.js`, 5 scenarios — the 3 trap scenarios + `taunt_struggle` /
     `pp_stall_struggle`, `gen3_bridge_struggle_resolve_v1`) → `bridge_test.rs`.
  2. **`gen3_struggle_activate_sideupdate_v1`** — a forced-Struggle mon emits `|-activate|<mon>|move:
     Struggle` (Struggle's `onModifyMove`) as a PER-SIDE `sideupdate` line, OWNER-ONLY, before the
     broadcast `|move|` batch (verified: the raw omniscient log emits it prefixed `pN\n…sideupdate`, so
     `getPlayerStreams` shows it only to the owner — like a `|request|`). `run_full_battle_bridge`
     injects it per-side after a Move commit whose active `must_struggle`. (The constructed protocol
     golden never runs a fully-out-of-PP mon.)

### THE BANKED SPEC QUEUE — 8 probe-settled specs, ALL SHIPPED (rounds 45-52)

Each is a re-runnable oracle in `harness/` whose header carries a SETTLED block: the draw model, the
exact emission forms, the edges, and the named way a naive implementation desyncs. **Read the probe
before implementing; do not re-derive from source.** They are listed with the trap that makes each
one non-obvious, because that trap is the reason the spec is worth more than the move.

| move(s) | probe | the trap |
|---|---|---|
| ~~Safeguard~~ **SHIPPED** | `probe_safeguard.js` | two Safeguards **TIE at residual order 4** → an extra shuffle, invisible to any single-side test; and a blocked SECONDARY still rolls its `random(100)` |
| ~~Recycle~~ **SHIPPED** | `probe_recycle.js` | the discriminator is WHICH primitive removed the item — `eatItem`/`useItem` set `lastItem`, `takeItem` does NOT, so Knock Off is **not** recyclable |
| ~~Fake Out~~ **SHIPPED** | `probe_fakeout.js` | gen-3 priority is **+1, not +3**; a CANCELLED action does not burn the first-turn gate but a CANT turn does, so `active_turns` is silently wrong |
| ~~Conversion / Conversion 2~~ **SHIPPED** | `probe_conversion.js` | `n == 1` **still draws**; the candidate list uses `types.names()` order, which matches neither our type chart nor the port's enum |
| ~~Torment~~ **SHIPPED** | `probe_torment.js` | joins the endTurn **DisableMove tie group** (n−1 draws); permanent, so a duration would add a phantom handler |
| ~~Imprison~~ **SHIPPED** | `probe_imprison.js` | the disable is **HIDDEN** — the request keeps `disabled:false` and gains `maybeDisabled`/`maybeLocked`; all-imprisoned SUBSTITUTES Struggle rather than rejecting |
| ~~Weather Ball~~ **SHIPPED** | `probe_weatherball.js` | the CATEGORY flips with the type (sandstorm is PHYSICAL) — **a test board with Atk==SpA and Def==SpD cannot see it**; reads `effective_weather` so Air Lock reverts it fully |
| ~~Skill Swap~~ **SHIPPED** | `probe_skillswap.js` | swapped abilities do **NOT** re-fire `onStart` (so re-running switch-in makes Trace draw where the sim does not) — but `onEnd` DOES fire, and that is the only draw it creates |


## The `ab_replay` SUBSEQUENCE SEED ANCHOR — what landed

**What landed.** `align_seed_subsequence` aligns the sim's per-decision seeds as a SUBSEQUENCE of the
port's checkpoints and the per-decision STATE/species/status/boost checks then run at the ALIGNED
pairs (ab_replay checks more per boundary than its bridge sibling, so aligning only the seed would
have left the state checks comparing unrelated boundaries). Verdicts still report the GOLDEN's
decision index, which is what a reader greps for in `battle.txt`.

**Held to round 20's bar, because a sloppy anchor makes the omniscient gate VACUOUS — far worse than
the artifact it removes.** `anchor_tests` is 10 cases and the NEGATIVES are the load-bearing half: an
**INJECTED EXTRA DRAW** still fails (a real desync permanently shifts every later seed VALUE, so the
sim's post-divergence seeds never reappear and the subsequence necessarily breaks), as do a missing
draw, a REORDERED pair (same values, wrong order), a port stream that ends early, and an empty port
stream. Gates: committed e2e golden still replays **220 ok / 0 diverged** through `ab_replay`; full
suite **75 binaries / 666 passed / 0 failed**; e2e md5 `3155eb…` UNCHANGED.

**NEW DIAGNOSTIC `POKESIM_DUMP_SEEDS=1`** prints BOTH per-decision seed lists. The anchor is only
sound if the port's checkpoint list really is a SUPERSET of the sim's, and that is now CHECKABLE on
any repro rather than assumed.

⚠️ **THE FINDING: `ab_41_9` is a CHECKPOINT-PLACEMENT artifact — the port's boundary lands exactly
ONE DRAW EARLY — and the anchor correctly does not reconcile it.** The port's RNG consumption is
CORRECT; there is no draw bug and no missing request.

**THE PROOF, from draw POSITIONS inside one decision** (`POKESIM_PRNG_TRACE=1` for the port,
`probe_repro_simtrace.js` for the sim). Decision 5 opens at the port's draw line 20 (the dec-4 seed):

| | |
|---|---|
| port's dec-5 checkpoint | line **29** → **9** draws consumed |
| the SIM's dec-5 seed | line **30** → the port's VERY NEXT draw |

and the sim's own trace carries the two values ADJACENT, in the same order:

```
[dec 6] draw#5  seed_before=12362,51825,42696,29605   <- the PORT's dec-5 checkpoint
[dec 6] draw#6  seed_before=58601,2165,55458,35804    <- the SIM's dec-5 checkpoint
```

Both engines therefore traverse the SAME seed values in the SAME order; only the pause point differs,
by one draw. The board is a forced-switch (dec 4, Swampert faints → Charizard) followed by a move
turn — the boundary bookkeeping shifts by one across that transition.

**WHY THE ANCHOR CANNOT ABSORB IT, and why that is correct.** The anchor matches sim seeds against
the port's **CHECKPOINT LIST**, and `58601` never appears there — it exists only in the port's DRAW
stream. Closing this class properly means anchoring against the DRAW stream instead, which is a
materially larger change carrying its own vacuity risk (the draw stream is dense, so a sloppy
draw-anchor would match almost anything). NOT attempted; recorded here as the next step for whoever
takes it, together with the warning.


## `--mode ourandom` — how the gen3ou-native generator is built

**EVERY INPUT IS SMOGON-DERIVED AND ALREADY COMMITTED** — no new data, no scraping:
`gen3_smogon_stats.json`'s per-species `usage` (which species appear) · `gen3_teammate_priors.json`
(the species×species joint, so teams are recognisable CORES not six unrelated mons) ·
`gen3_move_priors.json` / `gen3_item_priors.json` / `gen3_ability_priors.json` /
`gen3_spread_priors.json`. Deliberately NOT `data/teams/gen3_species_priors.json` — that is
POOL-derived, and the whole point is independence from the pool.

**LEGALITY IS SHOWDOWN'S VERDICT, NOT A REIMPLEMENTATION.** Every generated team goes through the
real `TeamValidator('gen3ou')`, so the banlist and every team-building clause are enforced by the
sim. Rejections are genuine gen-3 breeding facts (`Tyranitar can't get its egg move combination
(dragondance, pursuit)`), reject-sampled away.

**COVERAGE IS DISCLOSED, NOT ASSUMED.** Move sampling is renormalized over ENGINE-MODELED moves, so
a generated battle ALWAYS plays to completion — no fail-loud, no truncated prefix. The cost is
exactly the renormalized mass: **1.33% of gen3ou move-slot prior mass** sits on the 88 unmodeled
moves, printed in the run banner by `describeCoverage()` every run. Closing it is a NAMED, finite
queue: `sandattack recycle confuseray safeguard conversion weatherball fakeout imprison present
skillswap torment eruption`.

**HIDDEN POWER — the one non-obvious mechanic, and it is solved in closed form.** HP is ~12% of
gen3ou move slots (47.9% of mons carry one), so dropping it would gut the generator, and gen-3 HP
derives BOTH type and base power from IVs. BP 70 requires the bit-1 sum to be 63 — i.e. bit 1 set
in EVERY IV — which is true of both 30 (`11110`) and 31 (`11111`). So restricting IVs to {30,31}
PINS BP at 70 automatically and the bit-0 pattern alone selects the type; `hpIvsForType`
brute-forces the 64 patterns once. **Verified 16/16 by RECOMPUTING type and BP from the emitted
IVs**, not by trusting the table. Because BP is pinned, `allowHiddenPower` is enabled for this mode
exactly as it is for `pool`.

**TWO BUGS THE REAL DATA SHAPES CAUGHT** (both would have been invisible to a source read):
`gen3_spread_priors.json` stores a **LIST** of `[nature, [hp,atk,def,spa,spd,spe], weight]` triples,
not a dict — reading it as a dict silently passes the ARRAY INDEX as the nature, which Showdown
rejects with `"24" is an invalid nature`; and all typed Hidden Powers validate as ONE move, so a set
may carry at most one (`<species> has multiple copies of Hidden Power <type>`).

**First results:** a 50-battle smoke is `GREEN-GATE PASS`, 0 divergences, **69 distinct species and
95 distinct moves** — against 34 species in 25 `pool` battles. Same `{packed, genSeed}` interface as
`makeRandbatsProvider`, so it drops into `gen_sim_bridge_diff.js` unchanged too.



---

## `ab_fuzz.js` — the driver, the three team modes, the replayer and the repro format

- **The driver** `harness/ab_fuzz.js` — per chunk (default 25 battles): generate/pick team
  pairs, drive the omniscient BattleStream to game-end via the e2e recorder (**REUSED, not
  copied** — `runBattle`/`emitBattle`/`isModeledMove`/ability/item predicates are
  `module.exports`ed from `gen_e2e_fuzz.js`, one source of truth; a direct
  `node gen_e2e_fuzz.js` still regenerates the e2e goldens byte-identically under a
  `require.main` guard), write the chunk in the SAME TAB golden format (SCEN/TEAM/INIT/DEC/END),
  replay it through `ab_replay`, tally per-battle verdicts, and save repro dirs. One stats line
  per chunk → `<out>/ab_fuzz.log` (ok/diverged/cumulative/kinds/battles-per-hour/coverage/
  adjustment rate). A chunk that errors logs + continues; SIGINT finishes the chunk then
  summarizes (second SIGINT aborts). Flags: `--mode randbats|random|pool` (default randbats),
  `--battles N` / `--hours H` (default: until killed), `--master-seed S` (default from time,
  ALWAYS printed → reproducible), `--chunk N`, `--out DIR` (default `harness/ab_fuzz_out/`),
  `--keep-chunks`. Battles that hit a forced-UNMODELED state are kept as comparable PREFIXES
  (ended=0/none — `run_full_battle` reproduces that exactly) instead of being dropped.
- **The modes.** `randbats` (default): Showdown's OWN gen3 random-battle generator
  (`Teams.generate('gen3randombattle', {seed})` — probe-verified deterministic under a gen5-style
  seed; backed by `dist/data/random-battles/gen3/teams`). Sets are adapted at the SET level to be
  port-replayable: unmodeled item → Leftovers; unmodeled ability → the species' modeled/no-op
  ability, else the TEAM is rejection-sampled (real gen3 species are ability-saturated → high
  disclosed rejection rate; rejection reasons tallied); missing nature → Hardy (neutral,
  stat-identical both engines — counted apart from adjustments); movesets/levels/EVs UNTOUCHED
  (the picker never PICKS an unmodeled move). Adjustment rate logged per chunk — adjusted teams
  are "randbats-derived". `random`: the MODELED-UNIVERSE generator — 6 species per team sampled
  from every gen3-dex species whose learnset ∩ modeled-move-universe ≥4 (≥1 modeled DAMAGING move
  forced per mon), modeled/no-op ability, modeled item, random nature/EVs/IVs, level 100, packed
  with the real `Teams.pack` (round-trip-verified). This is the coverage multiplier (the first
  smoke: **195 species / 146 distinct moves exercised** — every universe species became active —
  vs the pool mode's 21 species / 37 moves) and the mode that flushes out modeled-predicate ↔
  engine drift. `pool`: the e2e's 22 filter-clean `data/teams/` teams with fresh seeds/choices.
- **The replayer** `src/bin/ab_replay.rs` (ADDITIVE — links the existing public lib APIs;
  zero engine change): parses the TAB chunk (non-panicking parser), replays each battle via
  `Battle::start_with_switchins` → `run_full_battle` at the recorded init seed, and emits ONE
  JSON verdict line per battle — `ok` or the FIRST divergence with a kind taxonomy
  (`seed` [draw bug] > `request` > `species` > `state` [hp/maxhp/fainted/left] > `status` >
  `boost` > `confusion` > `spikes` > `firstmover`, plus per-battle `start_error`/`init_seed`/
  `decision_count`/`ended`/`winner`), decision index, and expected-vs-got. Engine PANICS are
  **caught** (`catch_unwind` + a message-capturing hook) and reported as `"verdict":"panic"` —
  the loop never dies. It also replays a saved repro DIR directly
  (`ab_replay <dir>` reads `<dir>/battle.txt`) — the repro→pin path. Sanity: the committed e2e
  golden replays `ok:220 diverged:0` through it.
- **Repros** `<out>/divergences/<runid>_<battleid>/`: `battle.txt` (a single-battle chunk —
  standalone forever, independent of generator drift) + `summary.json` (mode, master seed,
  battle/init/choose seeds, packed teams, choice tokens, first-divergence
  kind/decision/expected/got/detail, replay_cmd). After an engine fix the same
  `ab_replay <dir>` must flip to `ok` — then pin it as a NAMED deterministic
  `tests/regression_test.rs` test per the edge-case→pin law.
- **Fault-injection proof (the tool catches bugs — verified 2026-07-03):** three one-at-a-time
  engine perturbations, each caught by a 6-battle pool run at a fixed master seed, each repro
  verified standalone-replayable, each restore verified byte-identical (`diff` vs a pristine
  copy) + the full suite green: (1) DROPPED DRAW (the unconditional end-of-turn Quick Claw
  `random_chance(1,5)` skipped) → 6/6 flagged **kind=seed** at dec 0; (2) STATE error (+1 damage
  on the non-absorbed apply path) → 6/6 **kind=state** (`hp=390` vs `hp=389`, seed matching);
  (3) WINNER flip on the game-end path → 5/6 **kind=winner** (`expected P2 got P1`; the 6th is a
  never-ended prefix battle — correctly still ok).


---

## `bridge_ab_fuzz.js` — the driver, the modes and the replayer

- **The driver** `harness/bridge_ab_fuzz.js` — per chunk: (1) drives a real in-process
  `BattleStream` + `getPlayerStreams` (the `local_sim_bridge.js` / `gen_bridge_capture.js` pattern)
  to game-end, picking random LEGAL + MODELED choices from a seeded choice-RNG, capturing BOTH
  per-side chunk streams + the ordered CMD stream; (2) **TRAPPING PROBES** — when a side's active is
  TRAPPED (Arena Trap / Magnet Pull / Shadow Tag, via the sim's `pokemon.trapped`), with `--trap-prob`
  (default 0.5) issues a REJECTED `switch` first (→ `|error|` + the `trapped:true`/`[Invalid]` re-request
  round) before the legal move; (3) drives `bridge_replay --ab` over the identical teams+cmds and diffs
  the Rust per-side chunks BYTE-FOR-BYTE with a first-divergence taxonomy (`preamble` / `perside` /
  `privacy`[HP-fold] / `request`[JSON] / `error`[trapped] / `chunk_count` / `panic`); (4) writes one
  stats line/chunk to `harness/bridge_ab_fuzz_out/bridge_ab_fuzz.log` + a self-contained
  standalone-replayable repro dir per divergence. `--debug`-free, no server.
- **Modes** (`--mode`, default `trapping`): **trapping** (a coordinated Arena-Trap/Magnet-Pull/Shadow-Tag
  matchup vs varied grounded/Flying/Levitate/Steel/Ghost foes + varied bench sizes incl. last-mon —
  the one mode where the port is bit-for-bit on the omniscient stream, so ALL divergences are genuine
  request/per-side-layer issues), **randbats**/**random**/**pool** (reuse `ab_fuzz.js`'s exported
  providers; genders pinned via `pinGenders`). `--format gen3customgame` (default, exact HP) or
  `gen3ou` (the OU framing + HP-privacy fold). One TAB golden format = `bridge_trapping_golden.txt`
  (SCEN/TEAM/INIT/CMD/CHUNK/END).
- **The replayer** `src/bin/bridge_replay.rs --ab` (ADDITIVE) — one JSON verdict per battle, panic-caught
  (a panic → `{"verdict":"panic"}`, never dies); it also replays a saved repro DIR directly
  (`bridge_replay <dir>` reads `<dir>/battle.txt`). Isolated build:
  `CARGO_TARGET_DIR=/tmp/pokesim_target_bridge` (NEVER the shared `target/` — the live `ab_replay`).


---

## `gen_sim_bridge_diff.js` — what landed, and the measured throughput

  randbats smoke reported **8/10 "diverged", ALL the same `return102` alias**, making the gate unusable.
  Now a documented request-DISPLAY residual is counted separately, filed under `<out>/allowlisted/`, and
  does not fail; exit is non-zero ONLY on a non-allowlisted divergence (the `ab_fuzz` convention). Keys
  mirror `bridge_replay.rs::classify_known_perside_residual`.
- **SAFETY: reconcile-then-compare.** The allowlist applies the documented transforms and requires the
  FULL lines to be byte-EQUAL after; ANY residual difference → not allowlisted → the gate FAILS. A
  co-occurring pair is allowed as a UNION.
- **⚠️ THE `gender-level-details` KEY IS DELIBERATELY NOT PORTED.** A blanket symmetric strip of
  `, L<n>` / `, <gender>` reconciles a genuine VALUE divergence (node `L84` vs rust `L83`, or M vs F)
  into a FALSE PASS — the exact way an allowlist turns a gate VACUOUS. It was written, caught by its own
  self-test, and REMOVED; the run then still went green under `return102-numeric-alias` alone, proving it
  was never needed. If a real presence-vs-absence residual appears, implement it PAIRWISE (strip only
  from the side that HAS the suffix) — never symmetrically. The other two keys are safe because they map
  an ALIAS to a CANONICAL form, so a real value difference still fails to reconcile.
- **`--selftest`** — 14 gate-integrity assertions: 10 allowlist ones whose negatives are load-bearing (a
  different move; an alias PLUS a residual pp difference; a LEVEL value difference; a GENDER value
  difference; a missing move; a non-request line; identical lines), plus 4 that pin the **switch-probe

  revert-verified 7.0 s → 181 s).
- **Throughput measured: ~600 battles/hr with `--persistent`, ~80/hr without** (each battle otherwise
  respawns a Node child that reloads the whole Showdown dist). **~96% of wall time is the per-write
  quiescence settle, not CPU** (`user` ≈ 2.3s per 59s wall) — so `--persistent` is mandatory for a soak,
  and the 40 ms settle is the remaining lever (still ~5x slower than `ab_fuzz`'s ~3000/hr).

**HONEST SCOPE (unchanged):** cross-side p1/p2 interleaving is not asserted (a Node scheduler artifact
the Python demux does not depend on); `__RECON__` is excluded (a real rust deferral; `resumeReseed` now works, `gen3_bridge_resume_reseed_v1`, so
reconstruction + search paths still require node); unmodeled moves fail loud, so "clean" is always
relative to the modeled universe.

**THE NEXT STEP THIS UNBLOCKS.** Running it at scale settles whether the round-26 `kind=seed` repros are
(a) offline-harness segmentation or (b) a draw-free legality bug — the question gating the `ab_replay`
subsequence anchor. A clean soak supports (a) and makes the anchor safe; request mismatches mean (b), and
the anchor would have HIDDEN them. **A further strengthening worth considering: parse both streams with
the real `src/poke_env` fork and diff the resulting battle STATE.** That is external consistency at the
OBSERVER's abstraction rather than the byte level — it would dissolve the `return102` class entirely
(poke-env resolves both forms) instead of allowlisting it on an assumption that is currently untested.



---

## The KNOWN-RESIDUAL ALLOWLIST + the GREEN GATE — the full E1 / A1 write-ups

*(The leaf keeps the mechanism, the six A1 clauses and the corpus gate; this is the
unabridged text, including the narrowing history, the repro ids and the GREEN results.)*

### The KNOWN-RESIDUAL ALLOWLIST + the GREEN GATE (`gen3_omniscient_byte_fuzz_v1`)

The byte fuzzer is a **GREEN GATE**: a NEW divergence fails loudly, while a DOCUMENTED, non-gen3ou-impacting
artifact is EXPLICITLY allowlisted (not silently ignored). Mechanism:
- **`ab_replay --protocol`** classifies a first-divergence protocol byte diff via
  `classify_known_residual(golden_framing, engine_framing, leads_speed_tie)` and adds an `"allowlisted":<reason>`
  field to the per-battle JSON verdict ONLY when the divergence FORM matches a documented residual; otherwise
  `allowlisted` is absent → the divergence counts as a hard failure. Entries are NARROW + STRUCTURAL (never a
  blanket "ignore protocol"):
  - **E1 `turn0-construction-speed-tie-attribution`** (R1) — the R1-SPECIFIC signature is a PURE PERMUTATION
    of identical-CONTENT framing lines at a construction speed-tie: the classifier takes the two FULL framing
    WINDOWS (all lines BEFORE the first `|turn|1`), sorts both, and allowlists ONLY IF `leads_speed_tie` AND
    the two windows are an IDENTICAL MULTISET (a pure reorder). The canonical R1 is a Zapdos-mirror both-Pressure
    lead where the port emits the two `-ability|…|Pressure|[silent]` lines in the OPPOSITE order (same multiset,
    different order — the unmodeled turn-0 construction speed-tie Fisher-Yates shuffle, the project-wide
    seed-convention deferral EVERY seed golden depends on). Both port paths (offline replay
    `run_full_battle_logged` AND the production bridge `run_full_battle_bridge` → `start_with_switchins`) share
    `event::run_start_switchins`, which falls back to a DETERMINISTIC side-order at a raw-Speed tie and draws
    nothing → ZERO production impact under `--use-bridge=rust` (seed=None: the port is the sole oracle). **This
    signature was NARROWED (2026-07-16, the Lens-1 green-gate-integrity fix)** from the prior coarse
    per-line-TYPE key (`line_type ∈ {-ability,-weather,-unboost}`), which SWALLOWED a content-DIFFERENT framing
    divergence at a mirror lead (a genuinely-wrong `[of]` target / wrong weather / a missing-or-extra line — a
    potential real bug). A CONTENT change now makes the multisets DIFFER → returns None → the gate FAILS
    (fault-injection-proven: injecting `-weather|Sandstorm|…|[of] p2a: Zapdos` in place of one `-ability` line of
    the R1 fixture no longer allowlists). The single-line weather-`[of]`-flip consequence is NOW covered by the
    NARROW **A1** key below (it is the SAME construction-window root, but a distinct structural form E1's
    permutation check deliberately does not match).
  - **A1 `turn0-construction-speed-tie-mirror-of-flip`** (2026-07-17, `classify_construction_mirror_of_flip`) —
    the single-line weather-`[of]`-FLIP on a same-species MIRROR lead (repros ab_4_8 / ab_11_20, BOTH formats):
    on a same-species mirror lead at a construction speed-tie, ONE framing `-weather`/`-ability` line's
    `[of] pNa: <name>` attribution flips between the two mirror active slots (golden `[of] p2a`, engine `[of]
    p1a`) because the unmodeled turn-0 construction speed-tie shuffle decides which same-species mon's Sand
    Stream fires last. NOT a pure permutation (the multiset DIFFERS — one line's `[of]` CONTENT changed), so E1
    returns None. Allowlisted ONLY when ALL SIX STRUCTURAL clauses hold (else None → the gate FAILS): (1) the
    construction speed-tie (`leads_speed_tie`); (2) the two equal-length framing windows differ in EXACTLY ONE
    line; (3) that line, in BOTH golden + engine, is a `-weather`/`-ability` framing line; (4) the two are
    byte-identical after stripping the trailing `|[of] pNa: <name>` clause (same weather/ability + `[from]`); (5)
    the two `[of]` targets are the two DIFFERENT active slots (one `p1a:`, one `p2a:`); (6) both `[of]` idents
    map — via the framing `|switch|pNa: <ident>|<Species>,…` details' species field — to the SAME species (the
    sibling mirror). A `[of]` to a NON-sibling / different-species mon, a different weather/ability prefix, a
    missing/extra framing line, or a non-mirror pair breaks a clause → None. seed=None-INVISIBLE (a cosmetic
    `[of]` tag on identical-species mirror mons → ZERO production impact under `--use-bridge=rust`; the port is
    the sole oracle at `seed=None`). **GATE-INTEGRITY PROVEN:** two cp-aside/mangled-golden injections (weather
    `Sandstorm`→`RainDance` breaks clause (4); a non-mirror `Tyranitar`→`Blissey` lead breaks clause (2)/(6))
    both replay to `diverged` with NO `allowlisted` field → the gate FAILS; plus 7 `#[cfg(test)]`
    `a1_allowlist_tests` in `ab_replay.rs` (incl. `clause6_wrong_of_to_a_real_different_species_mon_fails` — the
    load-bearing "wrong-[of]-to-a-real-different-mon → must FAIL"). Corpus fixture
    `27_construction_mirror_weather_of.txt` (tagged). **GREEN both formats** (master-seed 424242, 300 battles
    each): 0 non-allowlisted, 4 allowlisted (2 R1 `turn0-construction-speed-tie-attribution` + 2 A1
    `turn0-construction-speed-tie-mirror-of-flip` = ab_4_8 + ab_11_20).
- **`ab_fuzz.js --protocol`** counts `allowlisted` SEPARATELY from `diverged`, reports allowlisted-by-reason,
  and **exits non-zero ONLY on a non-allowlisted `diverged`/`panic`/`parse_error`** (a non-protocol run stays
  a hunter, exit 0). Allowlisted repros are saved under `<out>/allowlisted/` (auditable + a fixture source);
  real divergences under `<out>/divergences/`.
- **`tests/byte_fuzz_corpus_test.rs`** (the `cargo test` gate) enforces "no NEW kinds": each fixture resolves
  to either `ok` (the 20 emission-form fixtures stay byte-clean) OR a `diverged` verdict whose `allowlisted`
  reason EXACTLY equals a `# ALLOWLIST <reason>` header the fixture is tagged with — the R1 fixture
  `21_construction_mirror_ability_of.txt` (`turn0-construction-speed-tie-attribution`) + the A1 fixture
  `27_construction_mirror_weather_of.txt` (`turn0-construction-speed-tie-mirror-of-flip`). So (i) a residual
  fixture that stops matching its reason FAILS,
  and (ii) nobody can add a silently-ignored divergence — every escape is a named allowlist entry backed by a
  tagged repro. FAULT-INJECTION PROVEN: stripping the R1 fixture's tag (→ an un-cataloged divergence) makes
  the corpus test FAIL (`REGRESSION … kind=protocol`), restored byte-identical.
