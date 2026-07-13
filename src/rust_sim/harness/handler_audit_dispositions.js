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

// Weathers. Ability-set weather is PERMANENT (duration 0); no weather MOVE is modeled.
for (const id of ['sandstorm', 'raindance', 'sunnyday', 'hail']) {
  add([cond(id, 'duration'), cond(id, 'durationCallback')],
    UNREACH('a 5-turn duration applies only to MOVE-set weather — no weather move is modeled; ability weather resolves duration 0 (permanent); the rock items are gen4'));
  add([cond(id, 'onFieldStart')], IMPL('event.rs::run_switch', 'the ability-sourced weather set (permanent) + |-weather| line'));
  add([cond(id, 'onFieldEnd')], UNREACH('weather never ENDS in the modeled universe — permanent ability weather, no clearing move; replacement re-sets without an end'));
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
