# Protocol emission — the per-phase line inventory

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

The port emits the byte-identical omniscient `|...|` stream as a side output of events that
already happened. This document is the per-phase inventory of WHICH lines are emitted, in what
order, and the deferral record. The leaf keeps the emit API, the OBSERVATION-ONLY guarantee, the
byte-differential gate and the `write_line` drop-in.

---

## Phase 1 emits

- **Phase 1 emits** (the high-frequency core): the battle-init framing (`|t:|` normalized /
  `|gametype|singles` / `|player|` / `|gen|3` / `|tier|` / `|rule|` / the blank `|` separator /
  `|teamsize|` / `|start` / the leads' `|switch|`), the turn/phase markers (`|turn|N`, `|upkeep`),
  `|move|` (+ `[miss]`/`[still]`), `|switch|` / `|drag|`, `|-damage|` (all HP variants + the
  residual `[from]` tags — `Sandstorm` / `brn`/`psn`/`tox` / `Spikes`), `|-heal|` (+ `[from] item:
  Leftovers`), `|faint|`, `|-crit|`, `|-supereffective|`, `|-resisted|`, `|-immune|`, `|-miss|`,
  and `|win|` / `|tie|`. **Line ORDER inside a move** = `|move|` → `|-supereffective|`/`|-resisted|`
  → `|-crit|` → `|-damage|` (verified vs the golden). The `|faint|` ORDER follows Showdown's
  `faintQueue` (the self-KO'd Explosion USER before its KO'd target) via `faint_emit_queue` —
  since fix-queue #4 that queue is DRAW-BEARING, not emission-only (`gen3_faint_queue_order_v1`:
  `process_faints` drains it unconditionally so corpse ability-`End` events fire in enqueue order). The `|turn|N+1` marker is emitted at the TOP of the NEXT turn's outer-loop
  iteration (AFTER the previous turn's `|upkeep` + any forced replacement it triggered — a
  residual-faint's replacement `|switch|` precedes `|turn|N+1`), mirroring `makeRequest('move')`.


## Phase 2 emits

- **Phase 2 emits** (weather / boost / status / volatile / side-condition): the switch-in
  framing ability lines — `|-ability|<lead>|Intimidate|boost` + `|-unboost|<foe>|atk|1`
  (immunity-gated) + `|-weather|<W>|[from] ability: <A>|[of] <lead>` (RECONSTRUCTED in
  `emit_framing`'s `emit_switchin_ability_lines`, faster-lead first, from the post-switch-in state
  the construction already resolved); the STATUS-move `|move|` ANNOUNCE (`run_status_move`'s top —
  a self-target move renders the USER, a foe/foeSide move the FOE ACTIVE; the lone top-level
  `[still]` did-nothing form is **Spikes at the 3-layer cap**); `|-status|` (+ `[from] move: Rest`
  from `run_rest`) / `|-curestatus|`+`[msg]` (`on_before_move`'s slp/frz wake) / `|cant|`
  (par/slp/frz/flinch, `on_before_move`); `|-boost|`/`|-unboost|` by the CLAMPED delta's sign
  (setup `self_boost_spec` + the secondary `apply_secondary_boost` — a into-cap delta 0 emits
  nothing); `|-weather|<W>|[upkeep]` (the end-of-turn tick, at the TOP of `apply_weather_chip`);
  `|-fail|` (Recover-at-full / Rest-at-full / Spikes-at-cap / a phaze-with-no-bench / a failed
  Protect stall roll; + `move: Substitute`+`[weak]` for the Substitute can't-afford fail vs the
  bare `move: Substitute` already-up fail); `|-sidestart|<side>|Spikes`; the Substitute
  `|-start|`/`|-end|` (up / break) + `|-activate|…Substitute|[damage]` (a SURVIVED absorb, INSTEAD
  of `|-damage|`) + the sub-cost `|-damage|`; the Protect `|-singleturn|` (success) +
  `|-activate|…Protect` (block); and Rest's `|-heal|<user>|<HP> slp|[silent]`. The weather chip
  `|-damage|` ORDER now reads the `each_event_shuffle` RETURN (the shuffled side order — a
  same-species speed-TIE Snorlax-vs-Snorlax mirror chips in the shuffle's permutation, revealed by
  the golden's `-damage` order), a state-/seed-INVARIANT read that only fixes the emitted order.


## Deferrals

- **NO scenario-level deferrals remain** (`DEFERRED_SCENARIOS` is EMPTY): `debug` (poke-env-ignored
  free-form sim text — a deliberate non-emit) stays filtered from BOTH sides. The two
  `status_para_and_boost_drop` / `secondary_status_flinch` scenarios (formerly deferred for a
  forced-replacement REQUEST-BOUNDARY resume "phantom") are now ASSERTED byte-exact — the blocker was
  NOT Seismic Toss (modeled bit-for-bit; `fixeddamage_test.rs` + FD1–FD4) and NOT a protocol gap but two
  fixes (`gen3_forced_replacement_resume_v1`): (1) the `run_full_battle` **reject-and-re-request gate**
  (`move_decision_is_legal`) — the "phantom" was really an INVALID scripted move slot after a
  replacement changed the active mon to one with FEWER moves (a 3-move Tyranitar → a 2-move Snorlax, then
  a scripted `move 3`), which the sim's `side.choose` REJECTS drawing 0 (probe
  `harness/probe_forced_replacement_queue.js`: the rejected choice → no `commitChoices`, no `turnLoop`);
  the port used to RUN a full turn for it (its own move no-op'd but the FOE's move + residual + Quick
  Claw drew) → the seed + line stream diverged. Now the invalid decision is SKIPPED (draw-free, re-pull
  the next) — VERIFIED zero-draw so the seed suite is byte-identical. (2) the standalone status-move
  **already-statused `|-fail|` emission** (`foe_status_move_fail`, see the turn.rs STANDALONE STATUS
  MOVES note) — a `|-fail|<target>|par` line the port omitted. The three `recover_and_rest` Struggle
  battles that were formerly per-battle-skipped now REPLAY byte-exact (`gen3_pp_tracking_v1` — PP
  tracking + the forced-Struggle substitution + the Choice-Band lock + the Struggle `|move|` /
  `|-damage|…|[from] Recoil|[of]` lines via `ProtocolBuilder::damage_of`); `unreplayable_move` catches
  NOTHING now. `-sethp`/`-cureteam`/`-setboost`/the Haze family/`-item`/`-prepare`/… land with their
  mechanic (still unbuilt).


## Phase 3 emits

- **Phase 3 emits** (`gen3_protocol_phase3_v1` — the formerly-deferred long tail, each byte-verified
  by a NEW capture scenario): the **taunt/disable residual `-end`s** (`|-end|<mon>|move: Taunt|[silent]`
  / `|-end|<mon>|Disable`) + the **Disable retro-edit forms** (a missed Disable gains the `[miss]` attr
  via the new `attr_last_move_miss`; the no-lastMove & already-disabled fails retro-edit to
  `|move|…|Disable||[still]` + `|-fail|<user>` — the re-Taunt fail likewise, fixing the old
  fail-on-target form); the **status-move `[miss]` retro-edit generally** (Hypnosis/WoW/Toxic — the
  pre-Phase-3 "status announces never carry [miss]" claim was a corpus artifact); the **Trace reveal**
  (`|-ability|<mon>|<Copied>|Trace|[from] ability: Trace|[of] <foe>`, lead + mid-battle re-trace); the
  **Flash Fire cycle** (`|-start|…|ability: Flash Fire` on the arm INSTEAD of `-immune`;
  `|-immune|…|[from] ability: Flash Fire` when already armed; `|-end|…|ability: Flash Fire|[silent]`
  on an ALIVE switch-out — a FAINT emits nothing, capture-proven); the **STATUS_IMMUNE block lines**
  (`|-immune|<target>|[from] ability: Limber/Water Veil/Immunity/Insomnia` — ONLY for a status-MOVE
  source, `try_set_status_impl(announce_immune_block)`; a blocked SECONDARY is silent, and the
  Immunity-phase Magma Armor block stays silent — no direct gen-3 freeze move exists to capture);
  the **Synchronize→Lum interleave** (`-status holder → -status source [from] ability: Synchronize
  [of] holder → -enditem [eat] → -curestatus [msg]` — the reflect's `-status` form now emitted by the
  recursive apply itself via `sync_reveal`, so the source's Lum tail lands after it; + the LumRest
  chain byte-verified); the **MID-BATTLE switch-in ability lines** (`emit_ability_start_lines`, shared
  with the framing): weather SET on a REAL change, `|-ability|<mon>|Pressure|[silent]`, and Intimidate
  three ways — `-unboost`, the Clear-Body blocked form `|-fail|<foe>|unboost|[from] ability: Clear
  Body|[of] <foe>`, and the Substitute case (NO `-ability`, just the gen3
  `|-hint|In Gen 3, Intimidate does not activate if every target has a Substitute.`); **Leech Seed**
  (`|-start|…|move: Leech Seed`, the residual `|-damage|…|[from] Leech Seed|[of] <seeder-active>` +
  `|-heal|<seeder-active>|…|[silent]`, the Grass `-immune`, the re-seed `[still]`+`-fail`, the
  `[miss]` form); **Splash** `|-nothing`; **Pay Day** `|-fieldactivate|move: Pay Day` (after the
  `-damage`, direct hits only); and the **Rest-at-full-HP `|-fail|<user>|heal`** detail token (a real
  byte bug the per-write gate caught — the Phase-2 corpus never realized a full-HP Rest). **Protocol
  review FINDINGS F1-F3 now CLOSED** (`gen3_protocol_phase3_review_v1`, byte-verified by 3 NEW capture
  scenarios — probes `harness/probe_f1_f2_f3_lines.js` / `probe_f2_ff_armed_miss.js` /
  `probe_levitate_miss.js`): **F1** a sub-blocked Leech Seed now emits `|move|<user>|Leech Seed||[still]`
  + `|-fail|<user>` (IDENTICAL to the already-seeded form — scenario `leechseed_into_substitute`);
  **F2** a MISSED `onTryHit`-class ABILITY immunity (Flash Fire / Water&Volt Absorb — POST-accuracy)
  now emits `[miss]`+`-miss`, NOT `-immune` (gen3 rolls accuracy BEFORE TryHit; Levitate + type-chart
  0× stay pre-accuracy `-immune` even on a would-be miss, probe-confirmed 40/40 — scenarios
  `flashfire_tryhit_miss` / `waterabsorb_tryhit_miss`); **F3** a LANDED Water/Volt Absorb now emits
  `|-immune|<t>|[from] ability: Water Absorb` (resp. Volt Absorb), not a plain `-immune`. STILL
  UN-EMITTED (uncapturable, documented at their sites): a sub-absorbed Pay Day's form, the
  WoW-arms-a-non-Fire-FF status-path `-start` (impossible in gen-3 OU), Own Tempo's confusion-move
  `-immune` (no confusion-inflicting volatile move modeled). Phase 3 also fixed two REQUEST-BOUNDARY
  gaps the blind-plan scenarios exposed (both pinned + revert-verified): **per-side choice acceptance**
  (`side.choose` holds one side's valid choice while the other's is rejected — the old whole-decision
  skip mis-mapped split-accept boundaries; pin
  `per_side_choice_acceptance_maps_split_accept_boundaries_to_the_sims_seeds` replays the golden's DEC
  rows at SEED level) and the **switch-to-fainted reject** (pin
  `rejected_switch_to_a_fainted_slot_is_skipped_draw_free`); the forced-replacement pull got the same
  per-side accumulation (a double replacement may arrive as two one-sided writes — the `write_line`
  pattern).



---

## The byte-differential gate's scenario corpus and the formatter unit gates

- **The byte-differential gate** (`tests/protocol_test.rs`): replays the capture golden through
  `run_full_battle_logged`, FILTERS both the golden's lines and the engine's output to the gated
  types (only `debug` + `error` + still-deferred-mechanic lines dropped from BOTH; `|t:|`
  normalized), and asserts BYTE-EQUALITY per line, in order, with a first-divergence panic. A
  TRUNCATED golden (no terminal `|win|`/`|tie|` — the capture hit a decision/turn cap mid-stall,
  e.g. `spikes_and_phaze/2`'s infinite Spikes-at-cap↔immune-EQ loop) is asserted as a byte-exact
  PREFIX of the longer engine output. **Result: 132 battles asserted, 19348 lines byte-equal**
  (up from the F1-F3 review's 114 / 16115; from Phase-2's 66 / 8721; 63 / 7223, 51 / 5630, and
  Phase-1's 30 / 1512), across ALL 22 scenarios — the 11 Phase-1/2 + the 8 Phase-3 (`taunt_lifecycle` /
  `disable_lifecycle` / `trace_switchin` / `flashfire_cycle` / `status_immune_lines` /
  `synchronize_lum_rest` / `midswitch_ability_lines` / `leechseed_splash_payday`) + the 3
  F1-F3-review scenarios (`leechseed_into_substitute` / `flashfire_tryhit_miss` /
  `waterabsorb_tryhit_miss`).
  **0 battles DEFERRED** (`DEFERRED_SCENARIOS` empty) + **0 battles skipped** (`unreplayable_move`
  catches nothing now that Struggle is modeled — `>= 3` battles now REPLAY a forced Struggle
  byte-exact). The
  formatters are ALSO pinned by deterministic unit gates in `protocol.rs` (`HpStatus` three
  variants, the `p1a:`/`p1:` split, `Cause` item/ability/move/bare, the `[still]`+`[miss]` `|move|`
  forms, the `-status` `[from] move: Rest` variant, `cant`, `-curestatus` `[msg]`, `-boost`/
  `-unboost` by sign, `-weather` set-vs-`[upkeep]`, the Substitute `-start`/`-end`/`-fail [weak]`
  forms, the Rest `[silent]` heal, the disabled-builder-emits-nothing invariant, and the Phase-3
  forms — `attr_last_move_miss`, `ability_silent`/`ability_traced`, `fail_unboost_from_ability`,
  `hint`/`nothing`/`fieldactivate_move`, the taunt/disable `-end`s, the `-fail|heal` detail).



---

## The emit API — the full formatter inventory

*(The leaf keeps the append-only / PRNG-free rule, the nickname-ident hazard and the
disabled-by-default guarantee; this is the unabridged bullet.)*

- **The emit API** (`protocol.rs`): `ProtocolBuilder` is an **append-only, PRNG-free** line
  buffer on `BattleState` (the `log` field) — with ONE sim-mirroring exception:
  `attr_last_move_still()`, the port of `Battle.attrLastMove('[still]')` (blank the last `|move|`
  line's target + append `|[still]`), for fail forms the sim itself decides RETROACTIVELY, after
  draws the announce preceded (today only Disable's onStart 0-PP-guard rejection,
  `gen3_disable_zero_pp_v1`). The engine pushes lines at hook points in `turn.rs`;
  the fiddly formatting lives in ONE place — `MonRef` (`p<N>a: <Name>`; a `SideRef` is
  `p<N>: <PlayerName>`), `HpStatus` (the three variants `x/y` / `x/y <status>` / `0 fnt`, the #1
  correctness point), `Cause` (`[from] item: <Item>` / `[from] ability: <A>` / `[from] move: <M>` /
  `[from] <bare>`), `STAT_TOKENS` (the `-boost`/`-unboost` stat names). Move display names come
  from the dex (`MoveData.name` — Title-Case spaced). A `MonRef`'s IDENT name is the mon's ON-FIELD
  NICKNAME (`turn.rs::display_name` = the packed set's `set.name`, ← `SpeciesData.name` only when the
  set has no nickname — mirroring Showdown's `Pokemon.name = set.name || species.name`), NOT the
  species: poke-env keys each mon by this `p<N>a: <nick>` token, so rendering the species there
  (e.g. `p1a: Zapdos` for a Zapdos nicknamed `Electhor`) makes poke-env fail to match the mon it
  already tracks and try to ADD a 7th — the localized/nicknamed-team overflow crash
  (`gen3_nickname_ident_v1`, pinned by `regression_test::nicknamed_mon_renders_nickname_in_every_ident_not_species`).
  The SPECIES name (`turn.rs::species_name`) lives ONLY in the `|switch|`/`|drag|` DETAILS field
  (`|switch|p1a: Electhor|Zapdos|<hp>`). Disabled by
  default (`ProtocolBuilder::new()` → off): `run_full_battle` never enables it, so the seed suite
  keeps an empty, cost-free buffer AND every emit hook is a no-op that touches nothing.
  `run_full_battle_logged` enables it, emits the framing, runs the SAME `run_full_battle`, and
  returns `(BattleOutcome, Vec<ProtocolLine>)`.
