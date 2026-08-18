// handler_audit_dispositions.js — the CURATED disposition table for the
// handler-completeness audit (consumed by dump_gen3_handlers.js; the same curated-table
// pattern as dump_gen3_mechanics.js's class overrides).
//
// Every (effect, hook) row in the enumerated reachable surface must resolve to exactly
// one of:
//   implemented           — the port prices it; `anchor` is a grep-able Rust symbol
//                           ("file.rs::symbol" — --audit verifies the file under src/
//                           contains the symbol).
//   noop_justified        — the handler exists but does nothing the port must model
//                           (display-only lines, base-data residue, a guard that is
//                           vacuous in the reachable universe). `reason` required;
//                           `probe` cited where the justification is non-obvious.
//   unreachable_justified — the trigger cannot occur in the port's reachable universe
//                           (a gen>=4 move/mechanic, a state the modeled move/ability
//                           set can never produce). `reason` required.
//   failloud_guarded      — the effect is deferred and the engine FAIL-LOUDs (or the
//                           e2e filter keeps it off the modeled path) rather than
//                           silently diverging. `reason` names the guard.
//
// KEEP HONEST: a disposition here is an AUDITED CLAIM. When the gate reports a
// fingerprint drift on a row, re-probe before re-accepting (the dist's semantics moved).
'use strict';

const path = require('path');
const fs = require('fs');

const REPO = path.resolve(__dirname, '../../..');
const itemsJson = JSON.parse(fs.readFileSync(path.join(REPO, 'data/pokemon/gen3_items.json'), 'utf8'));

const IMPL = (anchor, note) => ({ disposition: 'implemented', anchor, ...(note ? { reason: note } : {}) });
const NOOP = (reason, probe) => ({ disposition: 'noop_justified', reason, ...(probe ? { probe } : {}) });
const UNREACH = (reason, probe) => ({ disposition: 'unreachable_justified', reason, ...(probe ? { probe } : {}) });
const GUARDED = (reason) => ({ disposition: 'failloud_guarded', reason });

// ─────────────────────────────────────────────────────────────────────────────
// EXACT per-row table. Key = `${kind}:${id}:${hook}`.
// ─────────────────────────────────────────────────────────────────────────────
const EXACT = {};
function add(keys, entry) { for (const k of keys) EXACT[k] = entry; }
const ab = (id, hook) => `ability:${id}:${hook}`;
const cond = (id, hook) => `condition:${id}:${hook}`;
const mv = (id, hook) => `move:${id}:${hook}`;
const it = (id, hook) => `item:${id}:${hook}`;

// ── ABILITIES ────────────────────────────────────────────────────────────────
// DMG_MOD pinch family: the gen3 fold is the onBasePower chain; the onModifyAtk/SpA
// PRIORITIES are gen4-mod residue (no fn resolves — the mod-chain law).
for (const id of ['blaze', 'overgrow', 'swarm', 'torrent']) {
  add([ab(id, 'onBasePower')], IMPL('turn.rs::resolve_bp_mods', 'pinch BP ×1.5 at 3·hp<=maxhp (gen3_item_mechanics_v1 P2)'));
  add([ab(id, 'onModifyAtkPriority'), ab(id, 'onModifySpAPriority')],
    NOOP('gen4-mod priority residue — the gen3 fold is the onBasePower pinch chain; no onModifyAtk/onModifySpA fn resolves (the mod-chain law)'));
}
add([ab('guts', 'onModifyAtk')], IMPL('turn.rs::resolve_atk_stat_mods', 'Atk ×1.5 whenStatused + the has_guts burn-halve suppression'));
add([ab('hugepower', 'onModifyAtk'), ab('purepower', 'onModifyAtk')], IMPL('turn.rs::resolve_atk_stat_mods', 'Atk ×2 (physical only)'));
add([ab('hustle', 'onModifyAtk')], IMPL('turn.rs::atk_direct_modify', 'the DIRECT this.modify(atk,1.5) pre-chain fold'));
add([ab('hustle', 'onSourceModifyAccuracy')], IMPL('turn.rs::effective_accuracy', 'acc ×3277/4096 for physical-TYPE moves (gen3_accuracy_pipeline_v1)'));
add([ab('marvelscale', 'onModifyDef')], IMPL('turn.rs::resolve_def_stat_mods', 'Def ×1.5 while the defender is statused'));
add([ab('thickfat', 'onSourceBasePower')], IMPL('turn.rs::defender_thick_fat', 'Ice/Fire sourceBasePower ×0.5 (defender handler on the attacker BP)'));
add([ab('thickfat', 'onSourceModifyAtkPriority'), ab('thickfat', 'onSourceModifySpAPriority')],
  NOOP('gen4-mod priority residue — the gen3 fold is onSourceBasePower; no onSourceModifyAtk/SpA fn resolves'));
add([ab('compoundeyes', 'onSourceModifyAccuracy')], IMPL('turn.rs::effective_accuracy', 'acc ×1.3 attacker chain'));
add([ab('sandveil', 'onModifyAccuracy')], IMPL('turn.rs::effective_accuracy', 'acc ×0.8 defender chain in sand'));
add([ab('sandveil', 'onImmunity')], IMPL('turn.rs::weather_immune', 'the sandstorm-chip immunity (AC5)'));

// STATUS_IMMUNE (gen3_status_immune_v1) + the Trace-route onUpdate cure.
for (const id of ['immunity', 'limber', 'insomnia', 'vitalspirit', 'waterveil']) {
  add([ab(id, 'onSetStatus')], IMPL('turn.rs::status_immune_of', 'the setStatus-phase block inside the already-drawn SetStatus event'));
}
add([ab('magmaarmor', 'onImmunity')], IMPL('turn.rs::status_immune_of', 'the immunity-phase frz block BEFORE the SetStatus event'));
for (const id of ['immunity', 'limber', 'insomnia', 'vitalspirit', 'magmaarmor', 'waterveil']) {
  add([ab(id, 'onUpdate')], IMPL('turn.rs::status_immune_on_update',
    'the Update-site cure — reachable ONLY via a statused mon TRACING the ability (gen3_statusimmune_onupdate_cure_v1, the A/B Trace-Porygon2 cluster)'));
}
add([ab('insomnia', 'onTryAddVolatile'), ab('vitalspirit', 'onTryAddVolatile')],
  UNREACH('gates the YAWN volatile only — Yawn is not a modeled move (fail-loud unmodeled status move)'));

// CRIT_IMMUNE / WEATHER_SPEED / WEATHER_NEGATE / RESIDUAL (batch 1).
add([ab('battlearmor', 'onCriticalHit'), ab('shellarmor', 'onCriticalHit')],
  IMPL('turn.rs::crit_immune', 'the drawn-then-overridden crit (B1, probe_critimmune_rng.js)'));
add([ab('chlorophyll', 'onModifySpe'), ab('swiftswim', 'onModifySpe')],
  IMPL('turn.rs::effective_speed', 'weather ×2 folded into the cached tie-shuffle speed (B2)'));
for (const id of ['airlock', 'cloudnine']) {
  add([ab(id, 'onStart')], IMPL('turn.rs::effective_weather', 'weather-effect suppression while the negater is active (B3)'));
  add([ab(id, 'onEnd')], IMPL('turn.rs::process_faints',
    'the onEnd eachEvent(WeatherChange) at the negater\'s faint — the faint-queue order fix (gen3_faint_queue_order_v1)'));
}
add([ab('speedboost', 'onResidual')], IMPL('turn.rs::run_residuals', 'ResidualAction::SpeedBoost at order 10 subOrder 3 (B4)'));
add([ab('raindish', 'onResidual')], IMPL('turn.rs::run_residuals', 'ResidualAction::RainDish +maxhp/16 in effective rain (B4b)'));
// FORECAST (`gen3_forecast_v1`, ROUND 35) — the Castform forme + TYPE swap.
add([ab('forecast', 'onWeatherChange')], IMPL('forecast.rs::forecast_each_event',
  'the whole handler (gates + forme + type + |-formechange|) at every WeatherChange site: the weather-set ' +
  'move arm, the mid-turn switch-in weather change, the two WEATHER_NEGATE onEnd sites (with the ending-' +
  'negater exclusion), and the previously-missing UNCONDITIONAL expiry WeatherChange (the T1 8-vs-7 fix)'));
add([ab('forecast', 'onStart')], IMPL('forecast.rs::forecast_weather_change',
  'the entrant singleEvent(WeatherChange) — run_switch fires it for the just-switched-in mon off the ' +
  'standing weather (probe S2 t3), and start_with_switchins applies the start-window equivalent post-hoc'));
add([ab('forecast', 'onSwitchInPriority')], IMPL('forecast.rs::forecast_weather_change',
  'ordering metadata for the sibling onStart (the −2 switch-in slot); PROBED IRRELEVANT in gen-3 singles ' +
  '(probe_r35_double_replacement.js: a double replacement emits switches in ENTRANT-SPEED order and each ' +
  'runSwitch fires its own singleEvent Start — the −2 never reorders anything observable; pin FC8)'));

// Batch-2 reactive classes.
for (const id of ['static', 'poisonpoint', 'flamebody', 'effectspore']) {
  add([ab(id, 'onDamagingHit')], IMPL('turn.rs::apply_contact_proc', 'the after-secondary contact-proc draw (B2-1/B2-2)'));
}
add([ab('roughskin', 'onDamagingHit')], IMPL('turn.rs::apply_contact_proc', 'the draw-free baseMaxhp/16 contact recoil (B2-3)'));
add([ab('soundproof', 'onTryHit')], IMPL('turn.rs::move_is_sound', 'sound-move immunity (B2-5)'));
add([ab('damp', 'onAnyTryMove')], IMPL('turn.rs::damp_holder', 'Explosion/Self-Destruct cancelled at TryMove (B2-4)'));
add([ab('damp', 'onAnyDamage')], UNREACH('the Aftermath guard — Aftermath is a gen4 ability, not in the gen3 universe'));
add([ab('suctioncups', 'onDragOut')], IMPL('turn.rs::drag_in', 'the phaze-arm Suction Cups gate — no drag, no sample draw (B2-6)'));
add([ab('synchronize', 'onAfterSetStatus')], IMPL('turn.rs::try_set_status', 'the source-aware status reflect (B2-7)'));
add([ab('liquidooze', 'onSourceTryHeal')], IMPL('turn.rs::apply_drain', 'the drain/leechseed heal REVERSAL (gen3_liquid_ooze_v1): the drainer/seeder takes the would-be heal as damage (|-damage|…|[from] ability: Liquid Ooze|[of]); apply_drain (drain moves) + apply_leech_seed (the leech residual); Dream Eater excluded (not a modeled drain move); DRAW-FREE'));
add([ab('wonderguard', 'onTryHit')], IMPL('turn.rs::run_move', 'the SE-only damage gate (gen3_wonder_guard_v1): a damaging move into a Wonder Guard holder CONNECTS only if runEffectiveness>0 (type-chart product >1.0) AND not type-immune, else -immune|[from] ability: Wonder Guard drawing only its accuracy roll; BYPASSED by Status / self-target / typeless(???/Struggle) / residuals; read-only gate, no new state; the firefang gen4-hint branch is unreachable (firefang not gen3-legal)'));

// Trapping + switch-in/out + trace.
for (const id of ['arenatrap', 'magnetpull', 'shadowtag']) {
  add([ab(id, 'onFoeTrapPokemon'), ab(id, 'onAnyTrapPokemon')].filter(() => true),
    IMPL('turn.rs::is_trapped', 'the switch-legality trap gate (gen3_trapping_v1 / B4-3 shadow tag)'));
  add([ab(id, 'onFoeMaybeTrapPokemon'), ab(id, 'onAnyMaybeTrapPokemon')],
    IMPL('turn.rs::is_trapped', 'the endTurn (Maybe)TrapPokemon handler gather — magnetpull\'s onAny* draws the 2-handler tie-shuffle'));
}
add([ab('intimidate', 'onStart')], IMPL('event.rs::intimidate_on_start'));
for (const id of ['drizzle', 'drought', 'sandstream']) {
  add([ab(id, 'onStart')], IMPL('event.rs::run_switch', 'switch-in weather set (permanent, ability-sourced); the Kyogre/Groudon orb guard is gen4-only'));
}
add([ab('naturalcure', 'onSwitchOut')], IMPL('turn.rs::execute_switch', 'the draw-free switch-out cure (gen3_natural_cure_v1)'));
add([ab('trace', 'onStart')], IMPL('event.rs::trace_on_start', 'the n=1 randomFoe sample + live ability copy (BR4)'));

// Batch-4 tail.
add([ab('truant', 'onBeforeMove')], IMPL('turn.rs::truant_turn', 'the priority-9 loaf cant (B4-1)'));
add([ab('truant', 'onResidual')], IMPL('turn.rs::run_residuals', 'the order-27 truant toggle residual'));
add([ab('truant', 'onSwitchIn')], IMPL('event.rs::run_switch', 'truant_turn armed `turn != 0` at switch-in'));
add([ab('innerfocus', 'onTryAddVolatile')], IMPL('turn.rs::apply_one_secondary', 'the flinch block at APPLY (the secondary random(100) still draws, B4-2)'));
add([ab('cutecharm', 'onDamagingHit')], IMPL('turn.rs::try_add_attract', 'the unconditional contact roll + gender-gated attract add (B4-4)'));
add([ab('colorchange', 'onDamagingHit')], IMPL('turn.rs::types_override', 'the mon_types choke-point type override (B4-5)'));

add([ab('shedskin', 'onResidual')], IMPL('turn.rs::run_residuals',
  'the order-10 subOrder-3 randomChance(33,100) cure-before-DoT residual (BR5; handler gathered unconditionally for the tie-shuffle)'));

// Absorbs / Flash Fire.
add([ab('voltabsorb', 'onTryHit'), ab('waterabsorb', 'onTryHit')],
  IMPL('turn.rs::apply_absorb_heal', 'the acc_hit-gated maxhp/4 absorb heal'));
add([ab('flashfire', 'onTryHit')], IMPL('turn.rs::apply_flash_fire_activation', 'the acc_hit-gated flash_fire arm (gen3_flashfire_v1)'));
add([ab('flashfire', 'onEnd')], IMPL('turn.rs::execute_switch',
  'REACHED routinely (Lens-1 corrected the old unreachable label): the sim fires singleEvent(End, ability) ' +
  'on EVERY switch-out AND faint (the gen3_cloudnine_end_v1 mechanism; probe: an armed FF switch-out prints ' +
  '|-end|...|ability: Flash Fire|[silent]). The body (removeVolatile(flashfire)) is draw-free and ' +
  'STATE-EQUIVALENT to the port volatile clear in execute_switch + process_faints (proven at scale: the e2e ' +
  'corpus has FF teams with switches, bit-for-bit). The [silent] -end line is now EMITTED on an alive ' +
  'switch-out (gen3_protocol_phase3_v1, byte-verified vs the flashfire_cycle capture; the FAINT path ' +
  'emits nothing — capture-proven). Formerly a documented un-emitted ' +
  'level-2 protocol nicety.'));

// SECONDARY_MOD + BOOST_IMMUNE.
add([ab('serenegrace', 'onModifyMove')], IMPL('turn.rs::apply_secondaries', 'secondary chance ×2 (incl. the King\'s Rock appended one)'));
add([ab('shielddust', 'onModifySecondaries')], IMPL('turn.rs::apply_secondaries', 'the filter-the-draw secondary suppression'));
for (const id of ['clearbody', 'whitesmoke', 'hypercutter', 'keeneye']) {
  add([ab(id, 'onTryBoost')], IMPL('event.rs::intimidate_on_start', 'the boost-drop immunity gate (Intimidate + secondary stat-drops)'));
}

// Own Tempo / Oblivious.
add([ab('owntempo', 'onTryAddVolatile')], IMPL('turn.rs::add_confusion', 'the confusion add block'));
add([ab('owntempo', 'onHit')], NOOP('display-only: the `-immune` line for a confusion-VOLATILE move (Confuse Ray etc. — no such move is modeled anyway); the block itself is onTryAddVolatile'));
add([ab('owntempo', 'onUpdate')], UNREACH(
  'cures a confusion the holder already has — unreachable: volatiles clear on switch-out, so a TRACE copy (the only ability-gain route) can never find a pre-existing confusion, and a direct add is blocked at onTryAddVolatile'));
add([ab('oblivious', 'onImmunity')], IMPL('turn.rs::try_add_attract', 'the attract immunity gate (batch-4 attract via Cute Charm)'));
add([ab('oblivious', 'onTryHit')], UNREACH('gates CAPTIVATE only — a gen4 move, not in the gen3 universe'));
add([ab('oblivious', 'onUpdate')], UNREACH(
  'removes an attract the holder already has — unreachable: attract clears when the holder switches out, so a TRACE copy can never find a pre-existing attract, and a direct add is blocked at the onImmunity gate'));

// NOOP_ABILITIES residues + guards.
add([ab('pressure', 'onDeductPP')], IMPL('turn.rs::pressure_extra', 'the foe-Pressure +1 PP deduction (gen3_pp_tracking_v1, VERIFIED −2 total)'));
add([ab('pressure', 'onStart')], IMPL('turn.rs::emit_ability_start_lines', 'the |-ability|<mon>|Pressure|[silent] switch-in reveal — EMITTED (gen3_protocol_phase3_v1, byte-verified; the omniscient stream carries the addSplit secret line)'));
add([ab('pickup', 'onResidualOrder'), ab('pickup', 'onResidualSubOrder')],
  NOOP('base-data residue: an out-of-battle pickup residual ORDER with NO function handler in the gen3 resolution'));
add([ab('stench', 'onModifyMovePriority')], NOOP('base-data residue: the gen9 stench-flinch onModifyMove is not in the gen3 resolution — only its priority number survives'));
add([ab('sturdy', 'onTryHit')], UNREACH('the OHKO-move immunity — the OHKO family is deferred (fail-loud + MOVE_ID_BLOCKLIST keeps it off the modeled path)'));
add([ab('sturdy', 'onDamagePriority')], NOOP('base-data residue: the gen5+ survive-at-1 onDamage is not in the gen3 resolution — only its priority number survives'));
add([ab('rockhead', 'onDamage')], NOOP(
  'negates recoil EXCEPT Struggle (the resolved handler: `if (this.activeMove.id !== "struggle") return null`) — Struggle is the ONLY recoil source in the modeled universe, so the handler never changes anything'));
add([ab('lightningrod', 'onFoeRedirectTarget')], NOOP(
  'redirects Electric moves in doubles — a singles battle has one target, nothing to redirect', 'probe_ability_batch1_noop_verify.js'));
add([ab('stickyhold', 'onTakeItem')], UNREACH('blocks Thief/Knock Off/Trick item removal — no item-removal move is modeled (MOVE_ID_BLOCKLIST)', 'probe_ability_batch1_noop_verify.js'));
add([ab('plus', 'onModifySpA'), ab('minus', 'onModifySpA')],
  IMPL('turn.rs::resolve_atk_stat_mods', 'the cross-field Plus↔Minus SpA ×1.5 chain member (gen3_plus_minus_v1)'));

// ── CONDITIONS ───────────────────────────────────────────────────────────────
// Major statuses.
for (const id of ['brn', 'par', 'slp', 'frz', 'psn', 'tox']) {
  add([cond(id, 'onStart')], IMPL('turn.rs::try_set_status', 'the status apply + |-status| emit (slp draws its random(2,6) duration at onStart)'));
}
add([cond('brn', 'onResidual'), cond('psn', 'onResidual')], IMPL('turn.rs::run_residuals', 'the maxhp/8 status DoT at order 10 subOrder 6'));
add([cond('tox', 'onResidual')], IMPL('turn.rs::tox_stage', 'the ramping stage chip at order 10 subOrder 6'));
add([cond('tox', 'onSwitchIn')], IMPL('turn.rs::stage_reset', 'the runSwitch-time toxic stage reset (TX pins)'));
add([cond('par', 'onModifySpe')], IMPL('turn.rs::effective_speed', 'the ×0.25 paralysis speed fold (quickfeet is gen4-only)'));
add([cond('par', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the priority-1 full-para randomChance(1,4) (magicguard is gen4-only)'));
add([cond('slp', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the draw-free counter decrement + wake; Early Bird double-decrements (earlybird read inline)'));
add([cond('slp', 'onSwitchIn')], UNREACH(
  'restores skippedTime — skippedTime is only ever incremented by a `sleepUsable` move (Snore / Sleep Talk, both MOVE_ID_BLOCKLISTed), so it is always 0'));
add([cond('frz', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the priority-10 randomChance(1,5) thaw BEFORE the defrost check'));
add([cond('frz', 'onModifyMove')], IMPL('turn.rs::run_move', 'the flags.defrost user self-thaw (gen3_defrost_v1 — Sacred Fire / Flame Wheel)'));
add([cond('frz', 'onDamagingHit')], IMPL('turn.rs::run_move', 'a landed Fire damaging move thaws the target (the fire-thaw at the DamagingHit region; sub-absorbed hits do not thaw)'));
add([cond('frz', 'onAfterMoveSecondary')], UNREACH('move.thawsTarget only — NO gen3 modeled move carries thawsTarget (the fire-thaw rides onDamagingHit instead)'));

// Volatiles.
add([cond('confusion', 'onStart')], IMPL('turn.rs::add_confusion', 'the random(2,6) duration; the lockedmove-fatigue branch is unreachable (lockedmove moves rejected)'));
add([cond('confusion', 'onBeforeMove')], IMPL('turn.rs::apply_confusion_self_hit', 'the priority-3 counter decrement + randomChance(1,2) self-hit'));
add([cond('confusion', 'onEnd')], NOOP('display-only |-end| confusion line (the port emits it at the counter expiry)'));
add([cond('flinch', 'duration')], IMPL('state.rs::flinch', 'the duration-1 volatile — reset at end of turn (clear_flinch)'));
add([cond('flinch', 'onBeforeMove')], IMPL('turn.rs::clear_flinch', 'the priority-8 draw-free cant'));
add([cond('substitute', 'onStart')], IMPL('turn.rs::run_status_move', 'the sub create (maxhp/4 cost, draw-free); the shedtail branch is gen9 cruft'));
add([cond('substitute', 'onTryPrimaryHit')], IMPL('turn.rs::absorb_into_sub', 'the sub absorb / break / no-carry + blocked status/stat-drop'));
add([cond('substitute', 'onEnd')], NOOP('display-only |-end| Substitute line (the port emits it at the break)'));
add([cond('leechseed', 'onStart')], IMPL('turn.rs::run_status_move', 'the |-start|<mon>|move: Leech Seed line — EMITTED at the plant (gen3_protocol_phase3_v1, byte-verified vs the leechseed_splash_payday capture)'));
add([cond('leechseed', 'onResidual')], IMPL('turn.rs::apply_leech_seed', 'the order-10 subOrder-5 drain/heal residual'));
// MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — CURSE / WISH conditions.
add([cond('curse', 'onStart')], IMPL('turn.rs::run_status_move', 'the |-start|<foe>|Curse|[of] <user> line — EMITTED at the ghost-curse lay (volatile_start_of)'));
add([cond('curse', 'onResidual')], IMPL('turn.rs::apply_curse', 'the order-10 subOrder-8 chip: the cursed foe loses floor(maxhp/4)/turn, draw-free'));
add([cond('wish', 'onEnd')], IMPL('turn.rs::apply_wish', 'the order-7 delayed heal floor(maxhp/2) at N+1 onto the slot occupant (|-heal|…|move: Wish|[wisher]); silent at full HP'));
add([cond('wish', 'duration')], IMPL('state.rs::wish_pending', 'the slot condition duration 2 (cast N → decrement to 1 → fire at 0 = end of N+1)'));
add([cond('protect', 'duration')], IMPL('state.rs::protected', 'the duration-1 volatile — cleared at turn top'));
add([cond('protect', 'onStart')], NOOP('display-only |-singleturn| Protect line'));
add([cond('protect', 'onTryHit')], IMPL('turn.rs::protect_blocks', 'the priority-3 TryHit block (after the attacker\'s accuracy roll)'));
add([cond('stall', 'duration'), cond('stall', 'counterMax'), cond('stall', 'onStart'), cond('stall', 'onRestart')],
  IMPL('state.rs::protect_counter', 'the gen3 stall counter 0→2→4→8 (counterMax 8) + duration-2 expiry (stall_duration)'));
add([cond('stall', 'onStallMove')], IMPL('turn.rs::run_protect', 'the consecutive-protect randomChance(1, counter); the counter>=256 branch is unreachable (counterMax 8)'));
add([cond('attract', 'onStart')], IMPL('turn.rs::try_add_attract', 'the gender gate INSIDE onStart (the CC roll drew unconditionally)'));
add([cond('attract', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the priority-2 -activate + randomChance(1,2) cant'));
add([cond('attract', 'onUpdate')], IMPL('turn.rs::run_update_items', 'the source-left clear (the attract onUpdate removal)'));
add([cond('attract', 'onEnd')], NOOP('display-only [silent] |-end| Attract line'));
add([cond('taunt', 'duration'), cond('taunt', 'onStart'), cond('taunt', 'onEnd')],
  IMPL('state.rs::taunt', 'the FIXED duration-2 taunt volatile (gen3_taunt_disable_v1); start/end lines emitted at the arm/expiry'));
add([cond('taunt', 'onDisableMove')], IMPL('turn.rs::move_usable', 'Status-category moves unselectable while taunted'));
add([cond('taunt', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the execution-time cant of a QUEUED status move'));
add([cond('disable', 'durationCallback')], IMPL('turn.rs::disable_move_event_shuffle', 'the ONE random(2,6) duration draw on a landed Disable'));
add([cond('disable', 'onStart')], IMPL('state.rs::disable', 'the willMove +1 duration store (gen3_taunt_disable_v1)'));
add([cond('disable', 'onDisableMove')], IMPL('turn.rs::move_usable', 'the disabled slot unselectable'));
add([cond('disable', 'onBeforeMove')], IMPL('turn.rs::run_move', 'the execution-time cant of the disabled queued move'));
add([cond('disable', 'onEnd')], IMPL('turn.rs::DisableDuration', 'the |-end|<mon>|Disable residual-expiry line — EMITTED (gen3_protocol_phase3_v1, byte-verified vs the disable_lifecycle capture)'));
add([cond('choicelock', 'onStart')], IMPL('state.rs::choice_locked_move', 'the lastMove lock arm (fails with no lastMove)'));
add([cond('choicelock', 'onDisableMove')], IMPL('turn.rs::move_usable', 'the non-locked slots unselectable while holding the Choice item'));
add([cond('flashfire', 'onStart')], IMPL('turn.rs::apply_flash_fire_activation', 'the flash_fire arm + |-start| line'));
add([cond('flashfire', 'onModifyDamagePhase1')], IMPL('turn.rs::flash_fire', 'the armed ×1.5 Fire fold (gen3_flashfire_v1 ModifyDamagePhase1)'));
add([cond('flashfire', 'onEnd')], IMPL('turn.rs::process_faints',
  'fires from the ABILITY\'s onEnd at every switch-out/faint — see ability:flashfire:onEnd; the volatile ' +
  'removal is state-equivalent to the port clear in execute_switch + process_faints.'));
add([cond('flashfire', 'onModifyAtkPriority'), cond('flashfire', 'onModifySpAPriority')],
  NOOP('gen4+ priority residue — the gen3 fold is onModifyDamagePhase1; no onModifyAtk/SpA fn resolves'));
add([cond('focusenergy', 'onStart')], IMPL('state.rs::focus_energy', 'the volatile arm (via a Lansat eat — the only gen3 route); the dragoncheer branch is gen9 cruft'));
add([cond('focusenergy', 'onModifyCritRatio')], IMPL('turn.rs::CRIT_MULT', 'the +2 crit-ratio fold (clamped 0..5) into the crit denominator'));

// Weathers. Ability-set weather is PERMANENT (duration 0); ALL FOUR weather MOVES are
// modeled 5-turn timed setters (`gen3_move_coverage_batch2_v1` rain/sun +
// `gen3_forecast_v1` hail/sandstorm) — the old "no weather move is modeled" reasons here
// were stale since batch 2 for rain/sun.
for (const id of ['sandstorm', 'raindance', 'sunnyday', 'hail']) {
  add([cond(id, 'duration'), cond(id, 'durationCallback')],
    IMPL('turn.rs::apply_weather_chip', 'the weather_turns 5-turn countdown (a MOVE set; ability weather is permanent duration 0; the rock items are gen4)'));
  add([cond(id, 'onFieldStart')], IMPL('event.rs::run_switch', 'the weather set (+ |-weather| line) — ability-sourced permanent OR the modeled_weather_set_move 5-turn arm'));
  add([cond(id, 'onFieldEnd')], IMPL('turn.rs::apply_weather_chip', 'the expiry branch: |-weather|none + the UNCONDITIONAL clearWeather eachEvent(WeatherChange) draw (the round-35 T1 fix)'));
}
add([cond('sandstorm', 'onFieldResidual'), cond('hail', 'onFieldResidual')],
  IMPL('turn.rs::run_residuals', 'the upkeep + effective-weather-GATED eachEvent(Weather) chip shuffle (a negater silences it)'));
add([cond('raindance', 'onFieldResidual'), cond('sunnyday', 'onFieldResidual')],
  IMPL('turn.rs::run_residuals', 'the UNCONDITIONAL eachEvent(Weather) tie-shuffle (the sun_rain_weather_turn_tie fix — fired off RAW weather)'));
add([cond('sandstorm', 'onWeather'), cond('hail', 'onWeather')],
  IMPL('turn.rs::weather_immune', 'the maxhp/16 weather chip (typed + Sand Veil immunities folded)'));
add([cond('raindance', 'onWeatherModifyDamage'), cond('sunnyday', 'onWeatherModifyDamage')],
  IMPL('damage.rs::weather', 'the Fire/Water ×1.5/×0.5 weather damage folds (hydrosteam is gen9 cruft)'));
add([cond('sunnyday', 'onImmunity')], IMPL('turn.rs::try_set_status', 'the sun-freeze immunity gate (gen3_sun_freeze_immunity_v1, FZ1)'));
add([cond('sandstorm', 'onModifySpDPriority')], NOOP('gen4+ SpD-boost priority residue — no onModifySpD fn resolves in gen3'));

// Spikes (the one gen3 side condition).
add([cond('spikes', 'onSideStart'), cond('spikes', 'onSideRestart')],
  IMPL('turn.rs::run_status_move', 'the layer set/increment (cap 3, a 4th FAILS) + |-sidestart| line — the spikes arm'));
add([cond('spikes', 'onEntryHazard')], IMPL('turn.rs::apply_entry_hazards',
  'the grounded switch-in chip ([_,3,4,6][layers]·maxhp/24, floored, min 1) at the gen3 runSwitch EntryHazard'));

// SCREENS (`gen3_move_coverage_batch2_v1`) — the Light Screen / Reflect side conditions.
for (const id of ['reflect', 'lightscreen']) {
  add([cond(id, 'onSideStart')], IMPL('turn.rs::modeled_screen_move',
    'the |-sidestart| line + the 5-turn counter set (the screen-set arm in run_status_move)'));
  add([cond(id, 'onSideEnd')], IMPL('turn.rs::run_residuals',
    'the |-sideend| line at expiry (the side-residual screen countdown in run_residuals)'));
  add([cond(id, 'duration')], IMPL('turn.rs::SCREEN_DURATION',
    'the fixed 5-turn duration (gen3 has no Light Clay → always 5)'));
  add([cond(id, 'durationCallback')], IMPL('turn.rs::SCREEN_DURATION',
    'the Light-Clay durationCallback resolves to 5 in gen3 (Light Clay is gen4)'));
  add([cond(id, 'onSideResidualOrder')], IMPL('turn.rs::run_residuals',
    'the side-residual TICK order (Reflect 1 / Light Screen 2 — the countdown decrement; the screens have no drawing residual handler)'));
  // The ModifyDamagePhase1 halving handler — the damage-calc fold PLUS the 2-screen tie-shuffle.
  add([cond(id, 'onAnyModifyDamagePhase1')], IMPL('damage.rs::modify_damage',
    'the ×0.5 physical(Reflect)/special(Light Screen) halving (crit-bypassed) — plus, when BOTH screens are up, the run_move ModifyDamagePhase1 tie-shuffle draw (two_tied_handler_shuffle)'));
}
add([cond('taunt', 'onResidualOrder'), cond('taunt', 'onResidualSubOrder')],
  IMPL('turn.rs::run_residuals', 'the duration-bearing taunt volatile\'s residual TICK slot (order 10 subOrder 15 — the fieldEvent Residual duration decrement; taunt itself has no onResidual fn)'));

// Format clauses.
add([cond('sleepclausemod', 'onSetStatus')], IMPL('turn.rs::side_has_sleeper', 'the clause block inside the drawn SetStatus event (clause formats only)'));
add([cond('freezeclausemod', 'onSetStatus')], IMPL('turn.rs::side_has_frozen', 'the second-freeze block (gen3_freeze_clause_v1 — the audit\'s HA2 fix)'));
add([cond('sleepclausemod', 'onBegin'), cond('freezeclausemod', 'onBegin')],
  NOOP('display-only: the |rule| banner line at battle start'));

// The dead `toxic` volatile (auto-added via the move's condition sub-object).
add([cond('toxic', 'duration'), cond('toxic', 'onSourceAccuracy'), cond('toxic', 'onSourceInvulnerability')],
  UNREACH('the gen6+ poison-type-Toxic never-miss volatile — NOTHING in the gen3 resolution adds it (the resolved gen3 toxic move carries no handler referencing it)'));

// ── ITEM exact rows (the class rules below cover the data-driven families) ──
add([it('choiceband', 'onAfterMove')], IMPL('state.rs::choice_locked_move', 'the post-move choicelock arm'));
add([it('quickclaw', 'onFractionalPriorityPriority')], IMPL('turn.rs::quick_claw',
  'the gen3 item entry is a bare priority number — the randomChance(1,5) lives in the gen3 action scripts, modeled as the per-turn QC draw'));
add([it('leftovers', 'onResidual')], IMPL('turn.rs::run_residuals', 'the maxhp/16 heal at order 10 subOrder 4'));
add([it('lumberry', 'onAfterSetStatus')], IMPL('turn.rs::try_set_status', 'LUM\'s IMMEDIATE eat (priority -1, AFTER Synchronize) at the setStatus tail'));
add([it('focusband', 'onDamage')], IMPL('turn.rs::focus_band_damage', 'the randomChance(1,10) on EVERY Damage event + survive-at-1 on a lethal MOVE hit'));
add([it('scopelens', 'onModifyCritRatio'), it('luckypunch', 'onModifyCritRatio'), it('stick', 'onModifyCritRatio')],
  IMPL('turn.rs::effective_crit_ratio', 'the CRIT_ITEM +N crit-ratio fold (gen3_crit_item_v1): Scope Lens +1, Lucky Punch +2 Chansey, Stick +2 Farfetch\'d — species-gated, draw-free (only the CRIT_MULT denominator index shifts)'));
// BOOST_RESTORE — White Herb (gen3_white_herb_v1): onStart scans boosts<0 → useItem; onUse
// setBoost(negatives→0) + -clearnegativeboost; fires from onAnyAfterMove / onAnySwitchIn /
// onResidual(29). All map to `white_herb_restore` (the scan + restore + emit), called at the
// after-move / switch-in stat-drop sites. DRAW-FREE.
add([it('whiteherb', 'onStart'), it('whiteherb', 'onUse'),
     it('whiteherb', 'onAnyAfterMove'),
     it('whiteherb', 'onAnySwitchIn'), it('whiteherb', 'onAnySwitchInPriority'),
     it('whiteherb', 'onResidual'), it('whiteherb', 'onResidualOrder')],
  IMPL('turn.rs::white_herb_restore', 'restore all NEGATIVE boost stages to 0 + consume, single-use (gen3_white_herb_v1); called at the after-move / switch-in stat-drop sites (apply_secondary_boost / apply_self_drops / run_switch / start_with_switchins); DRAW-FREE'));
add([it('whiteherb', 'onAnyAfterMega')],
  UNREACH('Mega Evolution is gen6+ — no AfterMega event fires in gen3'));
add([it('kingsrock', 'onModifyMove')], IMPL('turn.rs::flinch_secondary', 'the appended trailing 10% flinch secondary (execution-derived move list)'));
add([it('brightpowder', 'onModifyAccuracy'), it('laxincense', 'onModifyAccuracy')],
  IMPL('turn.rs::effective_accuracy', 'the DIRECT defender acc multiply (gen3_accuracy_pipeline_v1, AC3)'));
add([it('souldew', 'onBasePowerPriority')],
  NOOP('base-data BP priority residue — Soul Dew\'s gen3 folds are the onModifySpA/onModifySpD stat chains; no onBasePower fn resolves'));

// ── MOVE exact rows (fn hooks + the odd declaratives) ────────────────────────
add([mv('brickbreak', 'onTryHit')], UNREACH(
  'removes Reflect/Light Screen before the hit — the screens are unmodeled SIDE CONDITIONS no modeled move can set, so there is never a screen to break (the move\'s damaging path is fully modeled)'));
add([mv('detect', 'onPrepareHit'), mv('protect', 'onPrepareHit')], IMPL('turn.rs::run_protect', 'the willAct + StallMove gate (first protect short-circuits draw-free)'));
add([mv('detect', 'onHit'), mv('protect', 'onHit')], IMPL('state.rs::protect_counter', 'the stall volatile (re)add on success'));
add([mv('disable', 'onTryHit')], IMPL('turn.rs::run_status_move', 'the no-lastMove draw-free fail (gen3_taunt_disable_v1)'));
add([mv('facade', 'onBasePower')], IMPL('turn.rs::run_move', 'the ×2-when-statused BP chain member (gen3_facade_v1 — id-gated at the facade site)'));
add([mv('jumpkick', 'onMoveFail'), mv('highjumpkick', 'onMoveFail')],
  IMPL('turn.rs::apply_jump_kick_crash', 'the crash on miss/protect-block: getDamage-with-crit+roll halved, clamp [1, floor(target.maxhp/2)] (gen3_jump_kick_crash_v1 — the audit\'s HA1 fix)'));
add([mv('leechseed', 'onTryImmunity')], IMPL('turn.rs::run_status_move', 'the Grass-type immunity (accuracy drawn, then -immune)'));
add([mv('moonlight', 'onHit'), mv('morningsun', 'onHit'), mv('synthesis', 'onHit')],
  IMPL('turn.rs::recovery_heal_amount', 'the weather-conditional heal fractions'));
add([mv('payday', 'onHit')], IMPL('protocol.rs::fieldactivate_move', 'the |-fieldactivate|move: Pay Day line — EMITTED after the -damage on a landed direct hit (gen3_protocol_phase3_v1; the coin scatter itself has no battle effect; the sub-absorbed form stays uncaptured/un-emitted)'));
add([mv('rest', 'onTry')], IMPL('turn.rs::run_rest', 'the already-asleep / full-HP fail gates'));
add([mv('rest', 'onHit')], IMPL('turn.rs::run_rest', 'the full heal + FIXED Sleep(3) self-sleep (draws-then-discards the slp random(2,6))'));
add([mv('splash', 'onTry')], UNREACH('the Gravity gate — Gravity is a gen4 pseudo-weather, not in the gen3 universe'));
add([mv('splash', 'onTryHit')], IMPL('protocol.rs::nothing', 'the |-nothing marker — EMITTED (gen3_protocol_phase3_v1, byte-verified; Splash stays the modeled draw-free no-op move)'));
add([mv('struggle', 'onModifyMove')], IMPL('turn.rs::run_move', 'the typeless ??? override (no STAB, hits Ghosts) — the struggle arm resolves move_type None'));
add([mv('struggle', 'recoil')], IMPL('turn.rs::run_move', 'STRUGGLE RECOIL: the gen-3 recoil:[1,4] on ACTUAL damage dealt'));
add([mv('struggle', 'noPPBoosts')], IMPL('turn.rs::must_struggle', 'Struggle is not a slot — no PP tracked/deducted (gen3_pp_tracking_v1)'));
add([mv('substitute', 'onTryHit')], IMPL('turn.rs::run_status_move', 'the already-subbed / not-enough-HP fail gates'));
add([mv('substitute', 'onHit')], IMPL('turn.rs::run_status_move', 'the maxhp/4 directDamage create cost'));

// MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — CURSE / WISH / BATON PASS move rows.
add([mv('curse', 'onModifyMove'), mv('curse', 'onTryHit')], IMPL('turn.rs::run_status_move', 'the type-conditional re-target: NON-GHOST → move.self={boosts:{atk:1,def:1,spe:-1}} + target self (the selfDrops random(100) + boost); GHOST-into-a-sub deletes both (does nothing → [still]+-fail). (The resolved dist splits the logic across onModifyMove + onTryHit; the port handles both in the curse arm.)'));
add([mv('curse', 'onHit')], IMPL('turn.rs::run_status_move', 'the GHOST branch: lay the curse volatile on the foe THEN pay floor(maxhp/2) HP (draw-free)'));
add([mv('curse', 'nonGhostTarget')], IMPL('turn.rs::run_status_move', 'the NON-GHOST self-redirect target (`self`) — the port renders the announce/boost on the USER'));
add([mv('curse', 'volatileStatus')], IMPL('state.rs::curse', 'the `curse` volatile marker laid on the foe (Some(source_side))'));
add([mv('wish', 'slotCondition')], IMPL('state.rs::wish_pending', 'the Wish slot condition set at cast (double-Wish fails [still], draw-free)'));
add([cond('wish', 'onResidualOrder')], IMPL('turn.rs::apply_wish', 'order 7 — the Wish heal fires BEFORE the sand chip (order 8) + all order-10 handlers'));
add([mv('wish', 'onTryHit')], NOOP('undefined in the resolved dist — never-miss target:self, no accuracy check; the cast draws nothing'));
add([mv('batonpass', 'onHit')], IMPL('turn.rs::run_status_move', 'the no-eligible-bench FAIL ([still]+-fail, NOT_FAIL); else set switch_flag + baton_pass_pending for the forced self-switch'));
add([mv('batonpass', 'selfSwitch')], IMPL('turn.rs::execute_switch', 'copyvolatile: the copyVolatileFrom pass of boosts + the copyable volatiles (substitute/leech-seed/confusion/curse) to the entrant, [from] Baton Pass'));

// MOVE-COVERAGE BATCH 4 (`gen3_move_coverage_batch4_v1`) — FOCUS PUNCH + PURSUIT (the
// beforeTurnCallback + switch-interrupt classes). `priority` / `secondaries` etc. fall to the
// moveRule; these hooks (the two conditions + focuspunch onTry + pursuit onModifyMove) are
// EXACT. (`beforeTurnCallback` / `basePowerCallback` are declarative callbacks the enumerator
// does not surface as `on*` handlers; the beforeTurnMove queue action + the ×2 BP live in
// turn.rs::execute_beforeturn_move / turn.rs::run_move.)
add([mv('focuspunch', 'onTry')], IMPL('turn.rs::run_move', 'the onTry cancel BEFORE accuracy iff the focuspunch volatile is lostFocus (|move|…||[still] + |cant|…Focus Punch); PP/lastMove already deducted'));
add([cond('focuspunch', 'onStart')], IMPL('turn.rs::run_move', 'the |-singleturn|<user>|move: Focus Punch line at the beforeTurnMove (draw-free)'));
add([cond('focuspunch', 'onHit')], IMPL('turn.rs::run_move', 'sets lostFocus when the user is hit DIRECTLY by a non-Status move (a sub-absorbed chip does not count)'));
add([cond('focuspunch', 'onTryAddVolatile')], IMPL('turn.rs::apply_secondaries', 'the focuspunch volatile BLOCKS a flinch (the flinch-secondary random(100) still draws; draw-then-block)'));
add([cond('focuspunch', 'duration')], IMPL('state.rs::focus_punch', 'the duration:1 volatile — cleared at turn-top (clear_flinch) / switch-out / faint; registers a NO_ORDER residual duration handler (the FP-mirror tie-shuffle)'));
add([mv('pursuit', 'onModifyMove')], IMPL('turn.rs::run_move', 'accuracy=true (never-miss) on the interrupt condition (beingCalledBack/switchFlag) — the pursuit_strike flag drives the never-miss + ×2 BP'));
add([cond('pursuit', 'onBeforeSwitchOut')], IMPL('turn.rs::execute_switch', 'the interrupt: on a VOLUNTARY (is_voluntary) switch-out of the pursued mon, cancelMove the pursuer + run Pursuit at ×2 never-miss BEFORE the swap (gen3_move_coverage_batch4_v1); Baton Pass/drag/faint-replacement do NOT intercept'));
add([cond('pursuit', 'duration')], IMPL('state.rs::pursuit', 'the duration:1 volatile on the target (Some(pursuer_uid)) — laid at the beforeTurnMove, consumed by the interrupt / cleared at turn-top / switch-out / faint'));

// MOVE-COVERAGE BATCH 4b (`gen3_move_coverage_batch4b_v1`): BEAT UP / THUNDER / WATER SPOUT.
add([mv('beatup', 'onModifyMove')], IMPL('turn.rs::run_beat_up', 'adds the beatup volatile + sets type ??? / category Special / allies (healthy non-statused party) / multihit; the port runs the per-strike loop with the stat swap'));
add([cond('beatup', 'onModifySpA'), cond('beatup', 'onModifySpAPriority')], IMPL('turn.rs::run_beat_up', 'the stat swap: attacker SpA REPLACED by the current ally dex baseStats.atk (event.modifier=1) — computed directly in the per-strike DamageContext'));
add([cond('beatup', 'onFoeModifySpD'), cond('beatup', 'onFoeModifySpDPriority')], IMPL('turn.rs::run_beat_up', 'the stat swap: defender SpD REPLACED by the target dex baseStats.def (event.modifier=1) — computed directly in the per-strike DamageContext'));
add([cond('beatup', 'duration')], IMPL('state.rs::beat_up', 'the duration:1 volatile — a NO_ORDER/subOrder-2 residual duration handler (the Beat Up mirror tie-shuffle); cleared at turn-top (clear_flinch) / switch-out / faint'));
add([mv('thunder', 'onModifyMove')], IMPL('turn.rs::run_move', 'the id-gated weather-accuracy mutation: effective rain => never-miss (skip the accuracy random(100)), sun => base 50, else base 70 (gen3_move_coverage_batch4b_v1)'));

// MOVE-COVERAGE BATCH 4c (`gen3_move_coverage_batch4c_v1`): HYPER BEAM (mustrecharge) /
// SOLAR BEAM (twoturnmove) / DOOM DESIRE + FUTURE SIGHT (futuremove). The three
// cross-turn conditions are enumerated explicitly in dump_gen3_handlers.js's
// ENGINE_CONDITIONS (standalone conditions, not `m.condition` sub-objects).
add([mv('hyperbeam', 'self')], IMPL('turn.rs::run_move', 'self.volatileStatus mustrecharge — applied DRAW-FREE on a SUCCESSFUL damaging hit (plain/sub-absorb/sub-break/target-KO; NOT miss/immune/Protect-block), |-mustrecharge| after the damage/sub line'));
add([cond('mustrecharge', 'onStart')], IMPL('turn.rs::run_move', 'the |-mustrecharge|<user> announce at the apply site (log.must_recharge)'));
add([cond('mustrecharge', 'onBeforeMove'), cond('mustrecharge', 'onBeforeMovePriority')], IMPL('turn.rs::run_move', 'the priority-11 recharge cant at the TOP of run_move (before every status handler — a par/slp locked user rolls/decrements NOTHING): |cant|…|recharge, clear must_recharge, ZERO draws, NO PP; removeVolatile(truant) is a no-op in the truant_turn toggle model (the order-27 toggle consumes the loaf — probed cadence)'));
add([cond('mustrecharge', 'onLockMove')], IMPL('state.rs::move_locked', 'the locked single-`Recharge` request (choice_is_legal Move(0)-only + the firm trapped switch-reject; bridge serialize_active/resolve_choice)'));
add([cond('mustrecharge', 'duration')], IMPL('turn.rs::run_residuals', 'the duration:2 volatile registers a NO_ORDER/subOrder-2 residual duration handler on the CAST turn residual (the HB-mirror tie-shuffle); the removal is the recharge cant, not the countdown'));
add([mv('solarbeam', 'onTryMove')], IMPL('turn.rs::run_move', 'the two-turn gate: CHARGE ([still]+-prepare, zero move draws, addVolatile twoturnmove) / SUN SKIP (effective_weather-aware: -anim + immediate 3-draw execution) / FIRE (removeVolatile solarbeam => charging=false, [from]lockedmove, acc-100 drawn)'));
add([mv('solarbeam', 'onBasePower')], IMPL('turn.rs::run_move', 'the rain/sand/hail BP-halving chainModify(0.5) — a draw-free BP-chain fold read at damage time, suppression-aware (gen3 DOES have the modern halving, probed rain 54 vs control 105)'));
add([cond('twoturnmove', 'onStart')], IMPL('state.rs::two_turn', 'TwoTurnMove {move_index, duration:2, charging:true} set at the charge (the solarbeam sub-volatile == charging)'));
add([cond('twoturnmove', 'onLockMove')], IMPL('state.rs::move_locked', 'the locked single-move fire request (choice_is_legal Move(0)-only + the firm trapped switch-reject; the queue build maps Move(0) to the locked slot)'));
add([cond('twoturnmove', 'onMoveAborted')], IMPL('turn.rs::run_move', 'an onBeforeMove cant clears two_turn entirely — THE CHARGE IS LOST (a fresh charge re-pays PP)'));
add([cond('twoturnmove', 'onEnd')], IMPL('turn.rs::run_residuals', 'the duration expiry removes the volatile (TwoTurnDuration 2→1→0); its removeVolatile(solarbeam) is a no-op when the beam fired (charging already false)'));
add([cond('twoturnmove', 'duration')], IMPL('turn.rs::run_residuals', 'the duration:2 volatile registers a NO_ORDER/subOrder-2 residual duration handler on BOTH the charge- and fire-turn residuals (the SB-mirror tie-shuffle; after a fire-turn KO the resumed tail residual cleans the linger)'));
add([mv('doomdesire', 'onTry'), mv('futuresight', 'onTry')], IMPL('turn.rs::run_future_move_cast', 'the cast: addSlotCondition futuremove (a double-cast fails with a bare |move|, zero draws, PP still deducted) + ONE random(16) getDamage snapshot (typeless, cast-time stats, willCrit false) + |-start|<caster>|<Name>; BEFORE the protect block (a cast-turn Protect does not block)'));
add([mv('futuresight', 'ignoreImmunity')], IMPL('turn.rs::run_future_move_cast', 'the strike is TYPELESS ??? (move_type None — no chart row, never immune), so the declarative ignoreImmunity is subsumed'));
add([cond('futuremove', 'onStart')], IMPL('state.rs::future_move', 'FutureMove {duration:3, damage, move_id, accuracy, source_side/uid} — the slot-keyed pending strike (the Wish precedent)'));
add([cond('futuremove', 'onResidual'), cond('futuremove', 'onResidualOrder')], IMPL('turn.rs::apply_future_move', 'the order-11 residual tick (gathered EVERY end-of-turn while pending, speed = the slot occupant — the FS-mirror tie; Wish 7 → sand 8 → order-10s → futuremove 11 LAST), duration 3→2→1→resolve'));
add([cond('futuremove', 'onEnd')], IMPL('turn.rs::apply_future_move', 'the resolve: skip iff the occupant fainted; |-end|…|move: <Name>, remove Protect, ONE accuracy roll, the STORED number fixed-damage-style (sub absorbs, Focus Band rolls), the two hitStepMoveHitLoop Updates with the in-loop faintMessages between (a resolve KO draws one tie-Update + defers the Quick Claw)'));

// MOVE-COVERAGE BATCH 2 status-cure `onHit` (`gen3_move_coverage_batch2_v1`).
add([mv('refresh', 'onHit')], IMPL('turn.rs::run_status_move', 'the Refresh self-cure arm (par/psn/brn cleared; none/slp/frz fail — draw-free)'));
add([mv('healbell', 'onHit')], IMPL('turn.rs::run_status_move', 'the Heal Bell whole-team cure (active + bench; SKIPS a Soundproof ally, draw-free)'));
add([mv('aromatherapy', 'onHit')], IMPL('turn.rs::run_status_move', 'the Aromatherapy whole-team clearStatus cure (no Soundproof gate, draw-free)'));

// ─────────────────────────────────────────────────────────────────────────────
// CLASS RULES (applied when no exact row matches).
// ─────────────────────────────────────────────────────────────────────────────

// Move volatileStatus → the arm that models it.
const MOVE_VOLATILE_ANCHOR = {
  protect: IMPL('turn.rs::run_protect'),
  substitute: IMPL('turn.rs::absorb_into_sub', 'the substitute volatile (create in run_status_move, absorb at onTryPrimaryHit)'),
  taunt: IMPL('state.rs::taunt', 'the taunt volatile (gen3_taunt_disable_v1)'),
  disable: IMPL('state.rs::disable', 'the disable volatile (gen3_taunt_disable_v1)'),
  leechseed: IMPL('turn.rs::apply_leech_seed'),
  // CONFUSE RAY (`gen3_confuse_ray_v1`) — the volatileStatus is `confusion`, applied by the
  // SHARED `secondaries.rs::add_confusion` (the same path Water Pulse & co use), which owns the
  // KO / already-confused / Own-Tempo gates and the random(2,6) duration draw. The move's own arm
  // in `status_moves.rs` adds only the two MOVE-LEVEL emissions a secondary never produces
  // (`[still]`+`-fail` when already confused, `-immune|…|confusion|[from] ability: Own Tempo`).
  // ⚠️ This map is keyed by the VOLATILE NAME (`m.volatileStatus`), not the move id — the other
  // entries just happen to share both spellings. Confuse Ray's volatile is `confusion`.
  // TORMENT (`gen3_torment_v1`) — a PERMANENT selection-time restriction folded into
  // `state.rs::move_usable` (it blocks `last_move`, read LIVE). Deliberately NO residual
  // duration handler: gen-3 torment has no `duration`/`onResidual`, and a phantom one would
  // tie the NO_ORDER/subOrder-2 protect/stall/flinch group.
  torment: IMPL('state.rs::move_usable', 'the torment volatile — blocks the LAST-USED slot, permanent until switch-out (gen3_torment_v1)'),
  confusion: IMPL('secondaries.rs::add_confusion', 'the confusion volatile via the shared add_confusion — the same path Water Pulse & co use, owning the KO / already-confused / Own-Tempo gates and the random(2,6) duration draw (gen3_confuse_ray_v1)'),
};

function itemRule(row) {
  const mech = itemsJson[row.id] || {};
  const h = row.hook;
  // Berry family — the ONE eatItem mechanism + the four effect classes.
  if (mech.berryEffect) {
    const cls = mech.berryEffect.class;
    if (h === 'onEat') {
      const a = { cure: 'turn.rs::berry_cure', heal: 'turn.rs::BE::Heal', pinch: 'turn.rs::BE::Pinch', pp: 'turn.rs::BE::PpRestore' }[cls];
      return IMPL(a || 'turn.rs::apply_berry_residual', `the ${cls}-berry eat effect (gen3_berry_trace_shedskin_v1)`);
    }
    if (h === 'onUpdate') return IMPL('turn.rs::berry_on_update', 'the Update-site eat (cure/PP berries)');
    if (h === 'onResidual') return IMPL('turn.rs::apply_berry_residual', 'the order-10 subOrder-4 threshold eat (heal <=1/2, pinch <=1/4, exact — BR6)');
    if (h === 'onTryEatItem') return NOOP(
      'the runEvent(TryHeal) guard before a heal-berry eat — NO TryHeal handler exists in the gen3 modeled universe, so the guard is vacuous (the eat always proceeds)');
  }
  // TYPE_BOOST family (stat fold or BP fold) + SPECIES_STAT + CHOICE stat mods.
  if (mech.typeBoost) {
    if (h.startsWith('onModifyAtk') || h.startsWith('onModifySpA')) {
      return IMPL('turn.rs::resolve_atk_stat_mods', 'the type-gated offensive stat chain (gen3_item_mechanics_v1)');
    }
    if (h.startsWith('onBasePower')) {
      return IMPL('turn.rs::resolve_bp_mods', 'the BP chain / direct-float fold (bows + incenses; the stat-fold members carry only the base-data BP priority residue, same anchor)');
    }
  }
  if (mech.statMods) {
    if (h.startsWith('onModifyAtk') || h.startsWith('onModifySpA')) {
      return IMPL('turn.rs::resolve_atk_stat_mods', 'the species-gated offensive stat chain');
    }
    if (h.startsWith('onModifyDef') || h.startsWith('onModifySpD')) {
      return IMPL('turn.rs::resolve_def_stat_mods', 'the species-gated defensive stat chain');
    }
  }
  return null;
}


// MOVE-COVERAGE BATCH 5 (`gen3_move_coverage_batch5_v1`): COUNTER / MIRROR COAT /
// ENDEAVOR (the reactive fixed-damage family) + SLEEP TALK. (The VARIABLE-BP family's
// `basePowerCallback` is a declarative callback the enumerator does not surface — the
// engine BP lives in turn.rs::variable_bp; their `priority`/`secondaries` fall to the
// moveRule.) The fixed-damage `damage` declaratives (seismictoss/nightshade/sonicboom/
// dragonrage) surfaced when the batch-5 blocklist un-shadowing admitted them — covered
// by the moveRule `damage` clause.
add([mv('counter', 'onTry'), mv('mirrorcoat', 'onTry')],
  IMPL('turn.rs::run_fixed_damage_move', 'the ZERO-DRAW un-armed fail (no volatile / slot===null): a bare |move| line, no -fail, PP already deducted — BEFORE the accuracy roll'));
add([cond('counter', 'onStart'), cond('mirrorcoat', 'onStart')],
  IMPL('turn.rs::run_full_battle', 'the order-5 beforeTurnMove lays the reactive volatile with a RESET record ({slot:null,damage:0} — prev-turn damage never counts); draw-free, no protocol line'));
add([cond('counter', 'onDamage'), cond('mirrorcoat', 'onDamage')],
  IMPL('turn.rs::record_reactive_hit', 'the priority −101 (LAST — post-Focus-Band) recorder: 2× each qualifying DIRECT foe Move hit (counter: Physical || bare hiddenpower; mirrorcoat: Special && !hiddenpower), OVERWRITING per hit (multihit → 2× the LAST strike); a sub-absorbed hit never fires the mon Damage event'));
add([cond('counter', 'duration'), cond('mirrorcoat', 'duration')],
  IMPL('state.rs::reactive', 'the duration:1 volatile — a NO_ORDER/subOrder-2 residual duration handler (the counter-mirror tie-shuffle); cleared at turn-top (clear_flinch) / switch-out / faint'));
add([cond('counter', 'onRedirectTarget'), cond('mirrorcoat', 'onRedirectTarget')],
  IMPL('turn.rs::run_fixed_damage_move', 'target `scripted` redirects at the recorded damager slot — in gen-3 SINGLES that always resolves to the CURRENT foe active (probed C2: a foe switch fails the counter announcing the NEW active), which is what the port targets'));
add([mv('endeavor', 'onTry')],
  IMPL('turn.rs::run_fixed_damage_move', 'the ZERO-DRAW `hp >= target.hp` fail (EQUALITY INCLUDED — probed 50v50 fails): target-form announce + |-fail|<user>, BEFORE the accuracy roll'));
add([mv('endeavor', 'onTryImmunity')],
  IMPL('turn.rs::run_fixed_damage_move', 'pokemon.hp < target.hp — the same compare as the onTry (the port gates once, before accuracy); the Normal→Ghost type immunity rides move_is_immune after the accuracy draw'));
add([mv('sleeptalk', 'onTry')],
  IMPL('turn.rs::run_status_move', 'usable ONLY while asleep — an awake/wake-turn use fails SILENTLY (the normal self-target announce, no [still], no -fail, zero draws); `comatose` is gen7+, unreachable'));
add([mv('sleeptalk', 'onTryHit')],
  IMPL('turn.rs::run_status_move', 'the choicelock gate (a PRIOR-turn lock → [still]+-fail BEFORE the sample; the lock this use just set does not count — the pre-move snapshot); ENCORE is modeled as of batch 6 (`gen3_move_coverage_batch6_v1`) via move_usable + the onOverrideAction execution override'));
add([mv('sleeptalk', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the pool build (slot order, !nosleeptalk && !charge — data-driven MoveData flags; NO pp/disabled filter) + ONE sample = random(n) even at n=1 + the 0-PP |cant|nopp| stop + the bare-useMove called run ([from] Sleep Talk, no PP, full draw chain)'));
add([mv('sleeptalk', 'sleepUsable')],
  IMPL('turn.rs::on_before_move', 'the slp handler prints |cant|slp then PROCEEDS for Sleep Talk (skippedTime++; a normal blocked cant resets it; the onSwitchIn restore lives in run_switch)'));
add([mv('sleeptalk', 'neverMiss'), mv('sleeptalk', 'ignoreImmunity')],
  IMPL('turn.rs::run_status_move', 'accuracy true (no draw) + the Status-default ignoreImmunity — the sleeptalk arm draws nothing but the sample'));

// MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`): the FINAL UNMODELED tail —
// ENCORE / DESTINY BOND / ENDURE / PERISH SONG / MEAN LOOK / SPIDER WEB / BLOCK /
// BELLY DRUM / CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP (+ the linked
// trapped/trapper conditions). Probe-settled by probe_batch6_{locks,field_trap,
// utility,dexfacts}.js; pinned MC79-MC98.
add([cond('encore', 'onStart'), cond('encore', 'durationCallback')],
  IMPL('turn.rs::run_status_move', 'the encore arm: acc-100 draw → protect block → already-encored fail (acc ONLY) → durationCallback random(3,7) → the onStart rejects (no lastMove / failencore / 0-PP lastMove, draws consumed) → stored = willMove ? rolled : rolled+1 (MC79/MC80)'));
add([cond('encore', 'onOverrideAction')],
  IMPL('turn.rs::turn_loop', 'a queued DIFFERENT move executes AS the encored move — the ENCORED slot PP deducts, the announce shows the encored move (MC79/EN7)'));
add([cond('encore', 'onDisableMove')],
  IMPL('state.rs::move_usable', 'every non-encored slot is un-selectable while the volatile is up (the request disabled shape; a switch stays legal)'));
add([cond('encore', 'onResidual'), cond('encore', 'onResidualOrder'), cond('encore', 'onResidualSubOrder'), cond('encore', 'onEnd')],
  IMPL('turn.rs::run_residuals', 'the order-10/subOrder-14 tick (ResidualAction::EncoreDuration): decrement + the 0-PP EARLY -end (MC82) + the expiry -end'));
// SKILL SWAP (`gen3_skill_swap_v1`).
add([mv('skillswap', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the skillswap arm: never-miss (zero draws), Protect blocks / Substitute does not (bypasssub), FAIL on Wonder Guard (gen-3\'s only failskillswap) or an identical ability pair. Emits ONE gen<=4 line `|-activate|<u>|Skill Swap|||[of] <t>` (two EMPTY fields, no -endability/-ability). The swapped-in abilities do NOT re-fire onStart (the sim gates those on gen>3), but BOTH outgoing abilities DO fire onEnd — the weather_negate WeatherChange (a random(0,2) at a cached-speed tie) and the armed flash_fire silent -end, which is the ONLY draw this move can create. The switch-out revert is free: execute_switch already restores set.ability.'));
// FAKE OUT (`gen3_fakeout_v1`).
add([mv('fakeout', 'onTry')],
  IMPL('turn/moves.rs::run_move', 'the first-turn gate: `active_move_actions > 1` short-circuits the move DRAW-FREE (before the accuracy roll) with the announce plus |-hint|Fake Out only works on your first turn out. — NO [still] attr, NO -fail, landed=false, and `once=false` so a repeat emits the hint EVERY time. PP is still paid (the sim deducts before onTry). The counter is MOVE ACTIONS, not turns: a CANT-ed turn increments it (the sim bumps activeMoveActions at the top of runMove, before BeforeMove) while a turn whose action was CANCELLED by gen-3 faint-cancels-all does not.'));
add([mv('fakeout', 'priority')],
  IMPL('turn/moves.rs::run_move', 'gen-3 Fake Out is priority +1 (NOT the modern +3) — read from the dex row by the ordinary action sort, no special case.'));
add([mv('fakeout', 'secondaries')],
  IMPL('turn/secondaries.rs::apply_secondaries', 'the chance-100 flinch secondary STILL rolls its random(100) — the ordinary secondary path, no special case.'));

// RECYCLE (`gen3_recycle_v1`).
add([mv('recycle', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the recycle arm: FAIL ([still]+bare -fail on the USER) if the mon already holds an item OR has no last_item; else clear last_item BEFORE restoring it and emit |-item|<u>|<Item>|[from] move: Recycle. NEVER-MISS so zero draws on both paths, landed=false. `last_item` is set ONLY by the eatItem/useItem sites (items.rs::eat_item, white_herb_restore) — never by takeItem (Knock Off / Thief / Trick), which is why a knocked-off item is not recyclable.'));
// TORMENT (`gen3_torment_v1`) — the condition's own hooks.
add([cond('torment', 'onStart'), cond('torment', 'onDisableMove')],
  IMPL('turn.rs::run_status_move', 'the torment arm applies the volatile (bare `|-start|<t>|Torment`, NOT `move: Torment`); the onDisableMove restriction is folded into state.rs::move_usable against a LIVE `last_move`, and it JOINS the endTurn DisableMove tie group (speed.rs::disable_move_event_shuffle) — the only draw torment introduces'));
add([cond('torment', 'onEnd')],
  IMPL('turn.rs::execute_switch', 'UNREACHABLE in gen 3: nothing removes the volatile mid-battle (no duration, no onResidual), and clearVolatile wipes it SILENTLY without firing End — so the port clears the flag on switch-out/faint and deliberately emits NO `|-end|<t>|Torment`. Emitting one would be a pure protocol divergence; the byte form is UNVERIFIED because no probe can reach it.'));
add([mv('encore', 'volatileStatus'), mv('encore', 'ignoreImmunity')],
  IMPL('turn.rs::run_status_move', 'the encore arm applies the volatile; Status-default ignoreImmunity (no type gate)'));
add([mv('destinybond', 'onPrepareHit'), mv('destinybond', 'volatileStatus'), cond('destinybond', 'onStart')],
  IMPL('turn.rs::run_status_move', 'the DB arm: a ZERO-draw cast; the re-cast draw-free self-removes+re-adds (the flag is idempotent) — |-singlemove| each cast, PP −1 each'));
add([cond('destinybond', 'onBeforeMove'), cond('destinybond', 'onBeforeMovePriority'), cond('destinybond', 'onMoveAborted')],
  IMPL('turn.rs::run_move', 'the window closes at the users next move ATTEMPT: priority −1 removal for any move != destinybond + the onMoveAborted removal at every cant site (MC84)'));
add([cond('destinybond', 'onFaint')],
  IMPL('turn.rs::process_faints', 'the mutual-faint chain: |faint| victim → -activate → the killers hp zeroed + drained in the SAME faintMessages worklist (a foe-Move KO only — the damage sites record destiny_bond_ko_by; residual/futuremove/sub-absorbed never trigger; MC83/MC85)'));
add([mv('endure', 'onPrepareHit'), mv('endure', 'stallingMove'), mv('endure', 'priority'), mv('endure', 'onHit'), mv('endure', 'volatileStatus'), cond('endure', 'onStart')],
  IMPL('turn.rs::run_protect', 'the endure arm rides the SHARED protect stallingMove machinery (willAct gate + randomChance(1,counter) 2→4→8 no-delete-on-fail; priority 4); success sets MonState::endure'));
add([cond('endure', 'onDamage'), cond('endure', 'onDamagePriority')],
  IMPL('turn.rs::endure_clamp', 'the priority −10 survive-at-1 clamp on every MOVE-effect Damage site (plain / fixed / multihit strike / futuremove resolve); residual damage is NOT clamped (MC86/MC87)'));
add([cond('endure', 'duration')],
  IMPL('turn.rs::run_residuals', 'the duration:1 volatile registers a NO_ORDER/subOrder-2 residual duration handler (the endure+stall intra-mon tie — ONE shuffle on every SUCCESS turn); cleared at turn-top (clear_flinch)'));
add([mv('perishsong', 'onHitField'), cond('perishsong', 'duration')],
  IMPL('turn.rs::run_status_move', 'the perishsong arm: getAllActive in SIDE order — Soundproof -immune (counted as a result), fresh counters Some(4), ONE -fieldactivate iff >=1 newly applied, the all-counted [still]+-fail, the >=1-immune SILENT re-cast; ZERO draws every branch'));
add([mv('haze', 'onHitField')],
  IMPL('turn.rs::run_status_move', 'the haze arm (gen3_haze_v1): ONE |-clearallboost line + getAllActive().clearBoosts() zeroes BOTH actives\' 7 boost stages incl. the user\'s own; DRAW-FREE, landed FALSE'));
// YAWN (`gen3_yawn_v1`) — the delayed-sleep move + condition. The CAST is DRAW-FREE; the sleep
// random(2,6) fires at the RESOLVE (the residual onEnd at order 10 subOrder 19), routed through
// the EXISTING try_set_status(slp) path.
add([mv('yawn', 'onTryHit')],
  IMPL('turn.rs::run_status_move', 'the yawn arm TryHit gates: Protect blocks (-activate Protect) > already-statused → [still]+-fail > sleep-immune (Insomnia/Vital Spirit via runStatusImmunity(slp)) → -immune|[from] ability > Substitute → [still]+-fail > add the yawn volatile; ALL DRAW-FREE (accuracy true → no accuracy draw)'));
add([mv('yawn', 'volatileStatus')],
  IMPL('state.rs::yawn', 'the `yawn` delayed-sleep volatile marker laid on the foe (Some((duration, source_uid)))'));
add([cond('yawn', 'onStart')],
  IMPL('turn.rs::run_status_move', 'the |-start|<t>|move: Yawn|[of] <source> line — EMITTED at the cast (volatile_start_of); DRAW-FREE'));
add([cond('yawn', 'onEnd'), cond('yawn', 'onResidualOrder'), cond('yawn', 'onResidualSubOrder')],
  IMPL('turn.rs::run_residuals', 'the RESOLVE (ResidualAction::Yawn at order 10 subOrder 19): |-end|…|move: Yawn|[silent] then trySetStatus(slp) via the EXISTING try_set_status path — so the sleep random(2,6) at RESOLVE + the gen3ou Sleep Clause + the SetStatus shuffle come for free; the (10,19) tie-shuffle in a yawn MIRROR (Y1/Y2/Y3)'));
add([cond('yawn', 'duration')],
  IMPL('turn.rs::run_residuals', 'the duration:2 volatile: decrements 2→1 (cast turn) then 1→0 (resolve → fires onEnd); the handler is gathered while yawn.is_some(); cleared on switch-out + faint'));
// TRICK (`gen3_trick_v1`) — the item-SWAP move. ONE accuracy draw then a DRAW-FREE swap. onTryImmunity
// = the Sticky Hold block; onHit = the swap / knocked-off / both-itemless fail.
add([mv('trick', 'onTryImmunity')],
  IMPL('turn.rs::run_status_move', 'the trick arm Sticky-Hold onTryImmunity gate (!target.hasAbility(stickyhold)): a Sticky-Hold target draws the accuracy then reports PLAIN |-immune|<foe> (NO [from] ability), NO swap'));
add([mv('trick', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the trick arm swap: yourItem=target.takeItem(source), myItem=source.takeItem(); FAIL ([still]+-fail, no swap) if EITHER side item_knocked_off (gen<=4 takeItem→false) OR both itemless OR the foe has a Substitute (no bypasssub); else swap the two items (TARGET new-item line first, then USER; -item/-enditem [silent] [from] move: Trick) + RELEASE both choice_locked_move; DRAW-FREE (TR1-TR5)'));
add([cond('perishsong', 'onResidual'), cond('perishsong', 'onResidualOrder'), cond('perishsong', 'onEnd')],
  IMPL('turn.rs::run_residuals', 'the order-12 (LAST) tick (ResidualAction::Perish): decrement + |-start|perish<d>; the 1→0 tick prints perish0 + zeroes HP with the DURATION-END `continue` (NO per-handler faintMessages — the mutual perish-out double faint processes at the tail → the gen-3 TIE, MC89)'));
add([mv('meanlook', 'onHit'), mv('spiderweb', 'onHit'), mv('block', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the trap-move arm: protect block → substitute block ([still]+-fail) → already-trapped fail → the linked trapped_by = Some(trapper uid); ZERO draws every branch (MC90/MC91)'));
add([cond('trapped', 'onStart'), cond('trapped', 'onTrapPokemon'), cond('trapped', 'noCopy'), cond('trapper', 'noCopy')],
  IMPL('turn.rs::is_trapped', 'the FIRM trap (trap_is_firm true — the Shadow-Tag request shape); noCopy FALSE → the Baton Pass snapshot PASSES trapped_by (the link re-points, MC91); the link ends when the trapper leaves ANY way (execute_switch source-left clear + the process_faints corpse clear)'));
// PARTIAL TRAP (`gen3_partial_trap_v1`) — the wrap family + the resolved `partiallytrapped`
// condition (base `data/conditions.ts` shadowed by the gen5 onStart/onResidual and the gen4
// durationCallback/order overrides). Probe-settled: probe_batch89_trap.js + probe_ptrap_edges{,2}.js.
add([mv('wrap', 'volatileStatus'), mv('bind', 'volatileStatus'), mv('firespin', 'volatileStatus'),
     mv('clamp', 'volatileStatus'), mv('whirlpool', 'volatileStatus'), mv('sandtomb', 'volatileStatus')],
  IMPL('turn.rs::is_partial_trap_move', 'the post-hit partial-trap arm in run_move: gated on !absorbed (a Substitute intercepts before runMoveEffects — no volatile, NO duration draw) + a LIVE target (addVolatile early-returns on hp 0) + no trap already present (addVolatile returns false, no -fail); sets MonState::partial_trap and emits |-activate|<t>|move: <M>|[of] <u> after the -damage'));
add([cond('partiallytrapped', 'durationCallback'), cond('partiallytrapped', 'duration')],
  IMPL('turn.rs::run_move', 'THE family\'s ONE new draw: the gen4 override random(3,7) (uniform over {3,4,5,6}; the gripclaw arm is dead in gen3) stored as PartialTrap::duration. The base `duration: 5` is fully shadowed by the callback, so it never reaches the engine'));
add([cond('partiallytrapped', 'onStart')],
  IMPL('turn.rs::run_move', 'the gen5-override onStart: |-activate|<victim>|move: <Move>|[of] <trapper>. Its boundDivisor branch is CONSTANT 16 in gen3 (no Binding Band / Grip Claw), so the divisor is not stored'));
add([cond('partiallytrapped', 'onResidual'), cond('partiallytrapped', 'onResidualOrder'),
     cond('partiallytrapped', 'onResidualSubOrder'), cond('partiallytrapped', 'onEnd')],
  IMPL('turn.rs::apply_partial_trap', 'the order-10 subOrder-9 residual: duration-- FIRST (at 0 the onEnd fires |-end|<m>|<Move>|[partiallytrapped] with the DURATION-END `continue` and NO chip — this WINS over the trapper-gone branch, probe O), else the trapper-gone release (|-end|…|[silent], no chip) else the floor(maxhp/16) chip |-damage|…|[from] move: <Move>|[partiallytrapped] (Focus Band draws its onDamage roll, never survives)'));
add([cond('partiallytrapped', 'onTrapPokemon')],
  IMPL('turn.rs::is_trapped', 'the FIRM trap (bare tryTrap() → trapped:true on the FIRST request, a rejected switch is [Invalid choice] with NO re-request — probe M); the `source?.isActive` guard is mirrored by re-reading the foe active\'s uid, and the gmax arm is gen8-only (dead in gen3)'));

add([mv('bellydrum', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the bellydrum arm: the FLOAT hp<=maxhp/2 gate as integer-exact 2*hp<=maxhp (262/524 fails, 263 succeeds — MC92) + atk>=6 + maxhp==1 fails; directDamage(floor(maxhp/2)) then the SET to +6 (-setboost)'));
add([mv('charge', 'volatileStatus'), cond('charge', 'onStart'), cond('charge', 'onRestart')],
  IMPL('turn.rs::run_status_move', 'the charge arm sets MonState::charge (onRestart re-adds, -start again); gen3 has NO +1 SpD'));
add([cond('charge', 'onBasePower'), cond('charge', 'onBasePowerPriority')],
  IMPL('turn.rs::run_move', 'the ×2 BP-chain fold for the next ELECTRIC move (MC93)'));
add([cond('charge', 'onAfterMove'), cond('charge', 'onMoveAborted'), cond('charge', 'onEnd')],
  IMPL('turn.rs::turn_loop', 'the post-run_move consumption: any executed/aborted move != charge removes it (-end [silent]); keyed on the OUTER queued move (a Sleep Talk turn consumes on sleeptalk; the pursuit-interrupt bare useMove does NOT — no runMove → no AfterMove; MC93/MC98)'));
add([mv('memento', 'boosts'), mv('memento', 'selfdestruct')],
  IMPL('turn.rs::run_status_move', 'the memento arm: protect/sub block (NO faint — ifHit); the −2/−2 drops via apply_secondary_boost (Clear-Body gated; the user faints even when blocked/floored) then the self-faint through the deferred-faint protocol (gen3 faint-cancels-all + no QC → the ZERO-draw landed turn, MC94)'));
add([mv('mimic', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the mimic arm: sub / no-lastMove / failmimic (data-driven MoveData::fail_mimic) / already-known fails ([still]+-fail, draw-free); the copy overlays the slot {pp: min(5, base), maxpp: calculatePP(copied,3)} via MonState::mimic_overlay; restore_mimic_overlay reverts on switch-out/faint (MC95)'));
// TRANSFORM (gen3_transform_v1, ROUND 33) — the copy-overlay move.
add([mv('transform', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the transform arm (pokemon.transformInto): fainted / already-transformed target fails ([still]+-fail, draw-free — the USER being transformed does NOT block, that guard is gen5+); the copy overwrites species/types/5 non-HP stats/ability/all 7 boosts/moveslots {pp: min(5, base), maxpp: calculatePP(copied, ppUps[i] || 0) — the 1-move randbats Ditto gets NO pp-ups on slots 1-3}/hpType+hpPower via MonState::transform, sets cached_speed to the HYBRID spreadModify(TARGET base, OWN set).spe that setSpecies leaves behind, re-keys the slot-keyed choicelock/disable/lastMove references, and emits |-transform|U|T; restore_transform_overlay reverts on switch-out/faint (TF1-TF7)'));
add([mv('painsplit', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the painsplit arm: protect/sub block; avg = floor((u+t)/2), EACH side clamped at its OWN maxhp (MC96); -sethp target [silent] then user'));
add([mv('psychup', 'onHit')],
  IMPL('turn.rs::run_status_move', 'the psychup arm: copies ALL 7 stages VERBATIM (zeros overwrite — MC97); NO protect flag (copies through a Protect); bypasssub'));
// SNATCH (gen3_snatch_v1) — the LAST gen-3 status move, the status-steal.
add([mv('snatch', 'volatileStatus'), mv('snatch', 'priority'), cond('snatch', 'onStart')],
  IMPL('turn.rs::run_status_move', 'the snatch-cast arm: priority +4 sets MonState::snatch DRAW-FREE + emits |move|U|Snatch|U + |-singleturn|U|Snatch (the onStart line); landed FALSE'));
add([cond('snatch', 'onAnyPrepareHit'), cond('snatch', 'onAnyPrepareHitPriority')],
  IMPL('turn.rs::run_status_move', 'the INTERCEPTION (gated on the data-derived MoveData::is_snatchable + the foe snatch volatile): removeVolatile FIRST → |-activate|…|move: Snatch|[of] FOE → DeductPP (draw-free no-op) → the snatcher useMove(stolen) via a recursive run_status_move + [from] Snatch → return null (the foe move does nothing); MC100-MC104'));
add([cond('snatch', 'duration')],
  IMPL('turn.rs::run_residuals', 'the duration:1 volatile registers a NO_ORDER/subOrder-2 residual duration handler (the SNATCH-mirror tie-shuffle — 8 draws vs the both-Splash control 7, MC104); cleared at turn-top (clear_flinch) / switch-out / faint'));
add([mv('bellydrum', 'ignoreImmunity'), mv('bellydrum', 'neverMiss'),
     mv('snatch', 'ignoreImmunity'), mv('snatch', 'neverMiss'),
     mv('block', 'ignoreImmunity'), mv('block', 'neverMiss'),
     mv('charge', 'ignoreImmunity'), mv('charge', 'neverMiss'),
     mv('destinybond', 'ignoreImmunity'), mv('destinybond', 'neverMiss'),
     mv('endure', 'ignoreImmunity'), mv('endure', 'neverMiss'),
     mv('meanlook', 'ignoreImmunity'), mv('meanlook', 'neverMiss'),
     mv('memento', 'ignoreImmunity'), mv('memento', 'neverMiss'),
     mv('mimic', 'ignoreImmunity'), mv('mimic', 'neverMiss'),
     mv('painsplit', 'ignoreImmunity'), mv('painsplit', 'neverMiss'),
     mv('transform', 'ignoreImmunity'), mv('transform', 'neverMiss'),
     mv('perishsong', 'ignoreImmunity'), mv('perishsong', 'neverMiss'),
     mv('psychup', 'ignoreImmunity'), mv('psychup', 'neverMiss'),
     mv('spiderweb', 'ignoreImmunity'), mv('spiderweb', 'neverMiss')],
  IMPL('turn.rs::run_status_move', 'never-miss (accuracy true — NO accuracy draw; encore is the one acc-100 batch-6 move and draws) + the Status-default ignoreImmunity (painsplit works on a Ghost — probed)'));

function moveRule(row, d3) {
  const h = row.hook;
  const m = d3.moves.get(row.id);
  if (h === 'secondaries' || h === 'secondary') {
    return IMPL('turn.rs::apply_secondaries',
      row.id === 'triattack'
        ? 'Tri Attack\'s ONE random(100) + brn/par/frz sample (special-cased)'
        : 'the <=1-col secondary shape isModeledMove admits (status/flinch/confusion/boosts/self-boosts)');
  }
  if (h === 'neverMiss') return IMPL('turn.rs::never_miss', 'accuracy === true — NO accuracy draw at all');
  if (h === 'damage') {
    return IMPL('turn.rs::fixed_damage_amount',
      'the fixed/level damage declarative (Seismic Toss/Night Shade level; Sonic Boom 20; Dragon Rage 40) — accuracy-only draw, no crit/damage roll (gen3_fixeddamage_v1; surfaced by the batch-5 blocklist un-shadowing)');
  }
  if (h === 'priority') return IMPL('turn.rs::move_priority', 'the action-order priority sort (draws only on a priority+speed tie)');
  if (h === 'critRatio') return IMPL('turn.rs::CRIT_MULT', 'the high-crit denominator table (critRatio 2 => 1/8)');
  if (h === 'ignoreImmunity') {
    return IMPL('turn.rs::run_status_move',
      'the Status-category resolved default (true) / explicit false (Thunder Wave respects Ground) — the port\'s status arm implements the gen3 status immunities from first principles');
  }
  if (h === 'status' && m && m.category === 'Status') {
    return IMPL('turn.rs::modeled_status_move', 'the standalone status-inflicting move arm (accuracy + try_set_status)');
  }
  if (h === 'status') return IMPL('turn.rs::apply_secondaries', 'a secondary-carried status');
  if (h === 'boosts') {
    // A declarative top-level `boosts` map is EITHER a PURE self-boost setup move
    // (target:self, positive — `selfBoosts`) OR a standalone FOE STAT-DROP status move
    // (target:normal, negative — `statDropBoosts`, gen3_move_coverage_batch2_v1). Route by
    // the move's target so the anchor is right.
    if (m && m.target === 'self') {
      return IMPL('turn.rs::self_boost_spec', 'the pure self-boost setup arm (gen3_moves.json selfBoosts, GIGO-proof lockstep)');
    }
    return IMPL('turn.rs::stat_drop_boosts', 'the standalone foe stat-drop arm (Screech/Charm/Metal Sound/… — gen3_moves.json statDropBoosts; accuracy roll + a draw-free boost, gen3_move_coverage_batch2_v1)');
  }
  // WEATHER-SET (`gen3_move_coverage_batch2_v1`) — Rain Dance / Sunny Day set a 5-turn timed
  // weather (never-miss, draw-free set; the WeatherChange tie-shuffle draws only on a speed tie).
  if (h === 'weather') return IMPL('turn.rs::modeled_weather_set_move', 'the 5-turn timed weather-set arm (Rain Dance/Sunny Day; setWeather fails into the same weather, overwrites a different one)');
  if (h === 'heal') return IMPL('turn.rs::recovery_heal_amount', 'the flat floor(maxhp/2) recovery arm');
  if (h === 'volatileStatus') {
    const e = MOVE_VOLATILE_ANCHOR[String(m && m.volatileStatus)];
    if (e) return e;
    return null;
  }
  if (h === 'forceSwitch') return IMPL('turn.rs::drag_in', 'the phaze arm (accuracy + protect/soundproof/suctioncups gates + ONE sample)');
  if (h === 'selfdestruct') return IMPL('turn.rs::pending_explosion_self_ko', 'the pre-hit unconditional self-KO (gen3_explosion_v1)');
  if (h === 'sideCondition') {
    // SCREENS (`gen3_move_coverage_batch2_v1`) — Light Screen / Reflect set a 5-turn SIDE
    // condition the damage calc reads (halve special / physical). Spikes is the entry hazard.
    if (row.id === 'lightscreen' || row.id === 'reflect') {
      return IMPL('turn.rs::modeled_screen_move', 'the 5-turn screen side-condition arm (Light Screen/Reflect set + the side-residual countdown + the damage-calc halving)');
    }
    return IMPL('turn.rs::apply_entry_hazards', 'the Spikes side-condition arm (layers + grounded entry chip)');
  }
  if (h === 'stallingMove') return IMPL('turn.rs::run_protect', 'the protect/detect stall machinery');
  // MOVE-COVERAGE BATCH 1 (`gen3_move_coverage_batch1_v1`): the DRAW-FREE (+ self-drop's ONE
  // random(100)) post-hit effects, admitted via MODELED_{RECOIL,DRAIN,SELFDROP,ITEM_REMOVAL,
  // RAPIDSPIN}_MOVES. A recoil/drain/self move NOT in a modeled set is unreachable
  // (isModeledMove rejects it), so a row here is for an ADMITTED member.
  if (h === 'multihit') {
    // The MULTI-STRIKE count (`gen3_move_coverage_batch7_v1`): a fixed integer (Double Kick /
    // Twineedle / Bonemerang 2) draws nothing; a `[2,5]` array draws ONE `sample([2,2,2,3,3,3,
    // 4,5])`. Each strike runs the normal damage path (crit + random(16) + the per-strike
    // secondary) + the per-strike eachEvent, stopping at the target faint. Triple Kick
    // (multiaccuracy) is NOT reachable (the picker never admits it → the engine fail-louds).
    return IMPL('turn.rs::run_multihit', 'the multi-strike count + per-strike loop (fixed / [2,5] sample); stops at the target faint');
  }
  if (h === 'recoil') {
    return IMPL('turn.rs::apply_recoil', 'the recoil family (Double-Edge/Take Down/Submission); Rock Head negates, draw-free');
  }
  if (h === 'drain') {
    return IMPL('turn.rs::apply_drain', 'the drain family (Absorb/Mega/Giga Drain/Leech Life); floor non-sub, ceil behind a sub, draw-free');
  }
  if (h === 'self') {
    // `move.self` is EITHER a self-DROP (Overheat/Superpower `{boosts:...}`) or Rapid Spin's
    // empty `{}` marker (its clear is the onAfterHit/onAfterSubDamage below).
    if (m && m.self && m.self.boosts) {
      return IMPL('turn.rs::apply_self_drops', 'the self stat-drop (Overheat/Superpower) + the gen3 selfDrops random(100)');
    }
    return NOOP('an empty `self: {}` marker (Rapid Spin) — its effect is the onAfterHit/onAfterSubDamage clear');
  }
  if (h === 'onAfterHit' || h === 'onAfterSubDamage') {
    // Knock Off / Thief / Covet (item removal) + Rapid Spin (hazard clear).
    if (row.id === 'rapidspin') {
      return IMPL('turn.rs::apply_rapid_spin', 'the Rapid Spin hazard/leech clear (onAfterHit + onAfterSubDamage — clears behind a sub too)');
    }
    if (row.id === 'knockoff' || row.id === 'thief' || row.id === 'covet') {
      return IMPL('turn.rs::apply_item_removal', 'the Knock Off/Thief/Covet item removal (Sticky Hold block + the gen3 itemKnockedOff gate)');
    }
    return null;
  }
  return null;
}

// Numeric order/priority metadata inherits its sibling handler's disposition.
const METADATA_SUFFIX = /(Priority|SubOrder|Order)$/;
function metadataBase(hook) {
  // onResidualOrder -> onResidual; onBeforeMovePriority -> onBeforeMove; etc.
  let base = hook.replace(METADATA_SUFFIX, '');
  return base;
}

function dispositionFor(row, d3) {
  const exact = EXACT[row.key];
  if (exact) return exact;
  if (row.kind === 'item') {
    const r = itemRule(row);
    if (r) return r;
  }
  if (row.kind === 'move') {
    const r = moveRule(row, d3);
    if (r) return r;
  }
  // Order/priority metadata: inherit the sibling handler's disposition.
  if (METADATA_SUFFIX.test(row.hook)) {
    const base = metadataBase(row.hook);
    if (base !== row.hook && base.startsWith('on')) {
      const sibling = dispositionFor({ ...row, key: `${row.kind}:${row.id}:${base}`, hook: base }, d3);
      if (sibling) {
        return {
          ...sibling,
          reason: `${sibling.reason ? sibling.reason + ' — ' : ''}(order metadata of the sibling ${base} handler)`,
        };
      }
    }
  }
  return null; // undispositioned — the dump fails loudly.
}

module.exports = { dispositionFor };
