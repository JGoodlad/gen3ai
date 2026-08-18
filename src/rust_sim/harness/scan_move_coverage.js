// scan_move_coverage.js — the ENGINE-TRUE move-coverage scan for the gen3ou team pool.
//
// GOAL: for the 722 valid gen3ou teams under data/teams/, determine PER MOVE whether
// the Rust port's ENGINE can EXECUTE it without a fail-loud (`is not modeled` / a panic),
// then rank the unmodeled moves by "# teams carrying" + a greedy set-cover order, and
// group them into build-able mechanic classes.
//
// COVERAGE ORACLE (NOT the e2e picker `isModeledMove`): the modeled sets are mirrored
// DIRECTLY from `src/turn.rs` — the engine's own truth. `isModeledMove` has false
// positives for this purpose (it rejects `hiddenpower` [the engine models typed HP] and
// the modeled fixed-damage family [the engine RUNS them]). A separate Rust harness
// (scan_move_coverage_probe) EMPIRICALLY confirms each classification by actually driving
// the move through the engine; this JS scan is the deterministic static map + the team
// tallies.
//
// Run: node src/rust_sim/harness/scan_move_coverage.js
//   env: SCAN_JSON=1 to emit machine-readable JSON to stdout (else a human report).

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams, Dex, TeamValidator } = require(path.join(PS, 'dist/sim'));

const ROOT = path.resolve(__dirname, '../../..');
const TEAMS_DIR = path.join(ROOT, 'data/teams');
const RUST_MOVES = path.join(ROOT, 'data/pokemon/gen3_moves.json');

const VALIDATE_FORMAT = 'gen3ou';
const dex3 = Dex.forFormat('gen3customgame');
const rustMoves = JSON.parse(fs.readFileSync(RUST_MOVES, 'utf8'));

// ── The ENGINE's TRUE modeled sets (mirrored from src/turn.rs) ───────────────
// Each is the exact Rust `matches!`/data-driven set. If turn.rs changes, update here.

// Standalone status-inflicting moves (`modeled_status_move`) PLUS the volatile-inflicting
// status moves that have their own engine arm. `confuseray` (`gen3_confuse_ray_v1`) is the
// latter kind: it lives OUTSIDE `modeled_status_move` (which maps only MAJOR statuses) and is
// dispatched by its own arm in `status_moves.rs`, so a reader checking only that fn would
// wrongly conclude it is unmodeled.
const MODELED_STATUS = new Set([
  'thunderwave', 'stunspore', 'glare', 'toxic', 'poisonpowder', 'poisongas',
  'willowisp', 'spore', 'sleeppowder', 'hypnosis', 'sing', 'lovelykiss', 'grasswhistle',
  'confuseray',
]);
// Self-boost SETUP moves — DATA-DRIVEN from gen3_moves.json `selfBoosts` (== engine's
// `self_boost_spec`, which reads `MoveData::self_boosts`).
const MODELED_SETUP = new Set(
  Object.keys(rustMoves).filter((id) => {
    const sb = rustMoves[id] && rustMoves[id].selfBoosts;
    return sb && typeof sb === 'object' && Object.keys(sb).length > 0;
  })
);
// Self-heal / recovery (`recovery_heal_amount` + `run_rest` + splash).
const MODELED_RECOVERY = new Set([
  'recover', 'softboiled', 'slackoff', 'milkdrink',
  'moonlight', 'synthesis', 'morningsun', 'rest', 'splash',
]);
// Protect / Detect (`run_protect`).
const MODELED_PROTECT = new Set(['protect', 'detect']);
// Entry hazard — Spikes (`run_status_move` spikes arm).
const MODELED_HAZARD = new Set(['spikes']);
// Phaze (`modeled_phaze_move`).
const MODELED_PHAZE = new Set(['roar', 'whirlwind']);
// Leech Seed (`run_status_move` leechseed arm).
const MODELED_LEECH = new Set(['leechseed']);
// Substitute (`run_status_move` substitute arm).
const MODELED_SUBSTITUTE = new Set(['substitute']);
// MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — Curse / Wish / Baton Pass,
// all MODELED bit-for-bit (`run_status_move`'s curse/wish/batonpass arms + apply_curse/
// apply_wish + the copyVolatileFrom snapshot in execute_switch).
const MODELED_BATCH3 = new Set(['curse', 'wish', 'batonpass']);
// Selection-restriction (`MODELED_RESTRICTION_MOVES` — taunt/disable).
const MODELED_RESTRICTION = new Set(['taunt', 'disable', 'torment']);  // gen3_torment_v1
// MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`) — the FINAL UNMODELED tail,
// all category-Status, MODELED bit-for-bit (the batch-6 arms in src/turn.rs +
// gen_movecoverage_batch6_golden.js + the MC79+ pins): Encore / Destiny Bond / Endure /
// Perish Song / Mean Look / Spider Web / Block / Belly Drum / Charge / Memento / Mimic
// / Pain Split / Psych Up.
const MODELED_BATCH6 = new Set([
  'encore', 'destinybond', 'endure', 'perishsong',
  'meanlook', 'spiderweb', 'block',
  'bellydrum', 'charge', 'memento', 'mimic', 'painsplit', 'psychup',
]);
// MOVE-COVERAGE BATCH 1 (`gen3_move_coverage_batch1_v1`) — the DRAW-FREE (+ self-drop's ONE
// random(100)) post-hit effects on a DAMAGING move: recoil / drain / self-drop / item-removal
// / rapid-spin. The engine now models these bit-for-bit (they RUN + apply the side-effect), so
// they are MODELED, not the stale MISMODELED the pre-batch-1 scan reported.
const MODELED_RECOIL = new Set(['doubleedge', 'takedown', 'submission', 'volttackle']);
const MODELED_DRAIN = new Set(['absorb', 'megadrain', 'gigadrain', 'leechlife']);
const MODELED_SELFDROP = new Set(['overheat', 'superpower', 'psychoboost']);
const MODELED_ITEM_REMOVAL = new Set(['knockoff', 'thief', 'covet']);
const MODELED_RAPIDSPIN = new Set(['rapidspin']);
// MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — the DRAW-friendly status-move
// classes: status-cure / weather-set / stat-drop / screens.
const MODELED_CURE = new Set(['refresh', 'healbell', 'aromatherapy']);
// + hail / sandstorm (`gen3_forecast_v1`, ROUND 35 — the last two C_WEATHER_SET members).
const MODELED_WEATHER = new Set(['raindance', 'sunnyday', 'hail', 'sandstorm']);
const MODELED_STATDROP = new Set(
  // DERIVED from `gen3_moves.json`'s `statDropBoosts`, mirroring how the harness's own
  // MODELED_STATDROP_MOVES is derived — so relaxing the extractor guard (`gen3_sand_attack_v1`,
  // which admitted sandattack/smokescreen/kinesis/flash) updates the census automatically instead
  // of leaving it reporting a move as unmodeled after it was modeled.
  Object.keys(rustMoves).filter((id) => {
    const sd = rustMoves[id] && rustMoves[id].statDropBoosts;
    return sd && typeof sd === 'object' && Object.keys(sd).length > 0;
  })
);
const MODELED_SCREEN = new Set(['lightscreen', 'reflect']);
// MODELED fixed-damage (`fixed_damage_amount` — engine runs these bit-for-bit).
// BATCH 5 (`gen3_move_coverage_batch5_v1`): counter / mirrorcoat / endeavor are MODELED
// (the reactive volatile + recorder / the delta), no longer deferred.
const MODELED_FIXED_DAMAGE = new Set([
  'seismictoss', 'nightshade', 'sonicboom', 'dragonrage', 'superfang',
  'counter', 'mirrorcoat', 'endeavor',
]);
// DEFERRED fixed-damage (`is_fixed_damage_move` true but no amount → fail-loud panic).
const DEFERRED_FIXED_DAMAGE = new Set([
  'psywave', 'fissure', 'horndrill', 'guillotine', 'bide',
]);
// MOVE-COVERAGE BATCH 4 / 4b / 4c / 5 damaging-side modeled sets (mirrored from the
// src/turn.rs id-gates — the pre-batch-5 scan STALELY classified these MISMODELED):
//   BATCH 4  — Focus Punch + Pursuit (the beforeTurnMove queue machinery).
//   BATCH 4b — Beat Up (the only modeled multi-strike) / Thunder (the weather-accuracy
//              onModifyMove) / Water Spout (variable BP, data bp 150).
//   BATCH 4c — Hyper Beam (mustrecharge) / Solar Beam (twoturnmove) / Doom Desire +
//              Future Sight (the slot-keyed future strike).
//   BATCH 5  — the bp-0 VARIABLE-BP family (`variable_bp`): Return / Frustration /
//              Flail / Reversal / Low Kick.
const MODELED_BATCH4 = new Set(['focuspunch', 'pursuit']);
const MODELED_BATCH4B = new Set(['beatup', 'thunder', 'waterspout', 'eruption']);  // gen3_eruption_v1
const MODELED_BATCH4C = new Set(['hyperbeam', 'solarbeam', 'doomdesire', 'futuresight']);
const MODELED_VARBP = new Set(['return', 'frustration', 'flail', 'reversal', 'lowkick']);
// MOVE-COVERAGE BATCH 7 (`gen3_move_coverage_batch7_v1`) — the generic MULTI-STRIKE family
// (`run_multihit`): the FIXED-2 trio + the variable [2,5] family. `triplekick` (the sole
// `multiaccuracy` carrier) stays an ENGINE fail-loud and is deliberately NOT here.
const MODELED_BATCH7_MULTIHIT = new Set([
  'doublekick', 'twineedle', 'bonemerang',
  'pinmissile', 'bulletseed', 'iciclespear', 'rockblast', 'barrage', 'cometpunch',
  'doubleslap', 'spikecannon', 'armthrust', 'furyattack', 'furyswipes', 'bonerush',
]);
// ROUND 32 (`gen3_partial_trap_v1`) — the partial-trap family (foe `volatileStatus:
// 'partiallytrapped'`), modeled bit-for-bit (the random(3,7) duration draw + chip + firm trap).
const MODELED_PARTIALTRAP = new Set(['wrap', 'bind', 'firespin', 'clamp', 'whirlpool', 'sandtomb']);
// ROUND 40 (`gen3_unmodeled_move_failloud_v2`) — the 16 silent-desync moves now FAIL LOUD at
// CONSTRUCTION (`state.rs::UNMODELED_FAILLOUD_MOVES`, kept in LOCKSTEP + mirrored in
// gen_e2e_fuzz.js::REJECT_MOVES). Checked FIRST in classifyDamaging: whatever sub-mechanic
// bucket below would match, the engine panics before any of it can run.
const FAILLOUD_CONSTRUCTION = new Set([
  'dreameater', 'fakeout', 'falseswipe', 'furycutter', 'iceball',
  'outrage', 'petaldance', 'rage', 'revenge', 'rollout', 'secretpower',
  'smellingsalts', 'thrash', 'uproar', 'weatherball',
]);
// Typed Hidden Power — the engine models these end-to-end (16 typed nums 355-370 + bare).
// The bare `hiddenpower` id in a packed team resolves to a TYPED variant per the mon's IVs;
// the engine runs it as an ordinary damaging move. So HP is MODELED (contra isModeledMove).
function isHiddenPower(id) { return id === 'hiddenpower' || id.startsWith('hiddenpower'); }

// ── The damaging-move EXECUTION gate (mirrors run_move's damaging path) ───────
// The engine executes ANY category-Physical/Special move that reaches the damage path
// as a plain single-hit fixed-BP calc. It does NOT fail-loud on recoil/drain/multihit/
// charge/variable-BP/selfDrops — it would SILENTLY MIS-MODEL them (wrong damage / missing
// a draw). So for coverage we split damaging moves into:
//   RUNS_FAITHFUL  — a plain fixed-BP damaging move the engine runs bit-for-bit.
//   RUNS_MISMODELED — the engine runs it (NO fail-loud) but the sub-mechanic (recoil/drain/
//                     multihit/variable-BP/2-turn/selfDrops/onModifyMove) is WRONG → a
//                     silent desync. FLAG for a probe / a real fix.
// This classification uses the RESOLVED gen3 dex move data.
function classifyDamaging(m, id) {
  // ROUND 40: the construction-time fail-loud wins over EVERY bucket below — the engine
  // panics in MonState::from_set before the move could run at all.
  if (FAILLOUD_CONSTRUCTION.has(id)) {
    return { cov: 'UNMODELED', mech: 'construction fail-loud (round 40)' };
  }
  // A move whose damage bypasses getDamage → fixed-damage routing.
  if (MODELED_FIXED_DAMAGE.has(id)) return { cov: 'MODELED', mech: 'fixed-damage' };
  if (DEFERRED_FIXED_DAMAGE.has(id)) return { cov: 'UNMODELED', mech: 'reactive-or-ohko-fixed' };
  if (m.ohko) return { cov: 'UNMODELED', mech: 'ohko' };

  // SNORE (`gen3_move_coverage_batch5_v1`): the other sleepUsable move — fail-loud in
  // the engine (its awake-use onTry fail + the asleep proceed are unbuilt).
  if (id === 'snore') return { cov: 'UNMODELED', mech: 'sleep-usable damaging (fail-loud)' };

  // The batch-4/4b/4c/5 modeled sets — checked BEFORE the generic MISMODELED buckets
  // (each carries a callback/flag shape the generic checks would stale-classify):
  if (MODELED_BATCH4.has(id)) return { cov: 'MODELED', mech: 'beforeTurn (batch 4)' };
  if (MODELED_BATCH4B.has(id)) return { cov: 'MODELED', mech: 'multi-strike/weather-acc/variable-BP (batch 4b)' };
  if (MODELED_BATCH4C.has(id)) return { cov: 'MODELED', mech: 'turn-spanning (batch 4c)' };
  if (MODELED_VARBP.has(id)) return { cov: 'MODELED', mech: 'variable-BP (batch 5)' };

  // The REMAINING variable-BP moves with a bp-0 data row (Eruption / Grass Knot-class):
  // `derive_category` classifies a bp-0 move as Status → run_status_move's fail-loud
  // guard PANICS → UNMODELED (honest fail-loud). A remaining variable-BP move with a
  // NON-ZERO placeholder bp reaches the damaging path and runs at the wrong flat BP →
  // MISMODELED (silent desync).
  // BATCH 7: the generic multi-strike family is MODELED (`run_multihit`); `triplekick`
  // (the sole multiaccuracy carrier, ALSO a basePowerCallback holder — checked BEFORE the
  // variable-BP buckets) is an explicit engine fail-loud, NOT a silent run.
  if (MODELED_BATCH7_MULTIHIT.has(id)) return { cov: 'MODELED', mech: 'multi-strike (batch 7)' };
  if (id === 'triplekick') return { cov: 'UNMODELED', mech: 'multiaccuracy (fail-loud)' };
  // ANY remaining bp-0 damaging move (basePowerCallback OR onModifyMove-derived BP —
  // Magnitude / Present) fail-louds: the engine's gen-3 `derive_category` classifies bp 0 as
  // Status → `run_status_move`'s guard PANICS (probe-verified for magnitude/present).
  if (!(m.basePower > 0) && !m.damage && !m.damageCallback) {
    return { cov: 'UNMODELED', mech: 'bp0 → status fail-loud' };
  }
  if (m.basePowerCallback) return { cov: 'MISMODELED', mech: 'variable-BP' };
  if (m.multihit) return { cov: 'MISMODELED', mech: 'multi-hit' };
  // ROUND 32: the partial-trap family (foe volatileStatus) is MODELED; any OTHER damaging
  // move carrying a foe volatileStatus would run with the volatile silently dropped.
  if (MODELED_PARTIALTRAP.has(id)) return { cov: 'MODELED', mech: 'partial-trap (round 32)' };
  if (m.volatileStatus) return { cov: 'MISMODELED', mech: `foe-volatile (${m.volatileStatus})` };
  // MOVE-COVERAGE BATCH 1 (`gen3_move_coverage_batch1_v1`): recoil / drain / self-drop /
  // item-removal / rapid-spin are now MODELED bit-for-bit (they RUN + apply the side-effect),
  // checked BEFORE the MISMODELED buckets below (a Dream Eater / Liquid Ooze member NOT in a
  // modeled set stays MISMODELED — see the tail checks). A Rock Head user of a modeled recoil
  // move is still MODELED (the negation is modeled too).
  if (MODELED_RECOIL.has(id)) return { cov: 'MODELED', mech: 'recoil (batch 1)' };
  if (MODELED_DRAIN.has(id)) return { cov: 'MODELED', mech: 'drain (batch 1)' };
  if (MODELED_SELFDROP.has(id)) return { cov: 'MODELED', mech: 'self-drop (batch 1)' };
  if (MODELED_ITEM_REMOVAL.has(id)) return { cov: 'MODELED', mech: 'item-removal (batch 1)' };
  if (MODELED_RAPIDSPIN.has(id)) return { cov: 'MODELED', mech: 'rapid-spin (batch 1)' };
  if (m.recoil || (m.struggleRecoil)) return { cov: 'MISMODELED', mech: 'recoil' };
  if (m.drain) return { cov: 'MISMODELED', mech: 'drain' };
  // The remaining charge/recharge family (Blast Burn / Frenzy Plant / Hydro Cannon /
  // Razor Wind / Sky Attack / Skull Bash / Fly / Dig / Dive / Bounce) FAIL-LOUDS in the
  // engine since batch 4c (the explicit panic in run_move) → UNMODELED, not a silent
  // MISMODELED.
  if (m.flags && (m.flags.charge || m.flags.recharge)) return { cov: 'UNMODELED', mech: '2-turn-charge (fail-loud)' };
  // FUTURE MOVES (Doom Desire / Future Sight) — a `futuremove` flag + an `onTry` that queues a
  // delayed strike 2 turns out. bp>0 + no charge flag → reaches the damaging path + runs as an
  // INSTANT hit (wrong: state + draw desync). MISMODELED (empirically RUNS, no fail-loud).
  if (m.flags && m.flags.futuremove) return { cov: 'MISMODELED', mech: 'future-move (delayed strike)' }; // none left (DD/FS modeled)
  if (m.self && (m.self.boosts || m.self.volatileStatus)) return { cov: 'MISMODELED', mech: 'self-drop/lock' };
  if (m.forceSwitch) return { cov: 'MISMODELED', mech: 'phaze-damaging' }; // no gen3 damaging phaze
  if (m.onModifyMove) return { cov: 'MISMODELED', mech: 'onModifyMove (acc/power mutate)' };
  if (m.beforeTurnCallback) return { cov: 'MISMODELED', mech: 'beforeTurn' }; // none left (FP/Pursuit/Counter/MC modeled)
  if (m.damageCallback || m.damage) return { cov: 'UNMODELED', mech: 'derived-fixed-damage' };

  // Secondary shape > 1 col (except Tri Attack) → the engine's fail-loud >1-col guard PANICS.
  const secs = m.secondaries || (m.secondary ? [m.secondary] : []);
  if (secs.length > 1 && id !== 'triattack') return { cov: 'UNMODELED', mech: 'multi-secondary (fail-loud)' };

  return { cov: 'MODELED', mech: 'plain-damaging' };
}

// Classify a category-Status move against the engine's modeled status-move sets.
function classifyStatus(m, id) {
  if (MODELED_STATUS.has(id)) return { cov: 'MODELED', mech: 'status-inflict' };
  if (MODELED_SETUP.has(id)) return { cov: 'MODELED', mech: 'self-boost-setup' };
  if (MODELED_RECOVERY.has(id)) return { cov: 'MODELED', mech: 'recovery/rest/splash' };
  if (MODELED_PROTECT.has(id)) return { cov: 'MODELED', mech: 'protect' };
  if (MODELED_HAZARD.has(id)) return { cov: 'MODELED', mech: 'entry-hazard-spikes' };
  if (MODELED_PHAZE.has(id)) return { cov: 'MODELED', mech: 'phaze' };
  if (MODELED_LEECH.has(id)) return { cov: 'MODELED', mech: 'leech-seed' };
  if (MODELED_SUBSTITUTE.has(id)) return { cov: 'MODELED', mech: 'substitute' };
  if (MODELED_RESTRICTION.has(id)) return { cov: 'MODELED', mech: 'taunt/disable/torment' };
  // MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — status-cure / weather-set /
  // stat-drop / screens, all MODELED bit-for-bit.
  if (MODELED_CURE.has(id)) return { cov: 'MODELED', mech: 'status-cure (batch 2)' };
  if (MODELED_WEATHER.has(id)) return { cov: 'MODELED', mech: 'weather-set (batch 2)' };
  if (MODELED_STATDROP.has(id)) return { cov: 'MODELED', mech: 'stat-drop (batch 2)' };
  if (MODELED_SCREEN.has(id)) return { cov: 'MODELED', mech: 'screen (batch 2)' };
  // MOVE-COVERAGE BATCH 5 (`gen3_move_coverage_batch5_v1`) — SLEEP TALK (the
  // move-sampler). MODELED per-move; team-level playability composes naturally: a
  // sleep-talker whose POOL carries an unmodeled move is blocked by THAT move's own
  // row (the called move bypasses no gate the pool member itself doesn't).
  if (id === 'sleeptalk') return { cov: 'MODELED', mech: 'sleep-talk (batch 5)' };
  // MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`) — the final tail.
  if (MODELED_BATCH6.has(id)) return { cov: 'MODELED', mech: `batch-6 (${id})` };
  // MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — Curse / Wish / Baton Pass.
  if (MODELED_BATCH3.has(id)) {
    const mech = id === 'curse' ? 'type-conditional curse (batch 3)'
      : id === 'wish' ? 'delayed-heal Wish (batch 3)'
      : 'volatile-transfer Baton Pass (batch 3)';
    return { cov: 'MODELED', mech };
  }
  // SNATCH (`gen3_snatch_v1`) — MODELED bit-for-bit (the interception + cast in
  // run_status_move; the DEDICATED golden + MC100-MC104 pins).
  if (id === 'snatch') return { cov: 'MODELED', mech: 'status-steal Snatch (snatch)' };
  // The BATCH 8/9 + later status singles — each MODELED bit-for-bit with its own golden:
  if (id === 'haze') return { cov: 'MODELED', mech: 'boost-reset Haze (gen3_haze_v1)' };
  if (id === 'yawn') return { cov: 'MODELED', mech: 'delayed-sleep Yawn (gen3_yawn_v1)' };
  if (id === 'trick') return { cov: 'MODELED', mech: 'item-swap Trick (gen3_trick_v1)' };
  // TRANSFORM (`gen3_transform_v1`, ROUND 33) — the copy overlay, category Status in gen3.
  if (id === 'transform') return { cov: 'MODELED', mech: 'copy-overlay Transform (gen3_transform_v1)' };
  // Everything else is a fail-loud in run_status_move / run_protect. Bucket by mechanic.
  return { cov: 'UNMODELED', mech: statusMechanic(m, id) };
}

// The gen3 mechanic bucket for an UNMODELED status move (the build-class hint).
function statusMechanic(m, id) {
  const S = (x) => id === x;
  if (S('wish')) return 'delayed-heal (Wish)';
  if (S('healbell') || S('aromatherapy')) return 'team-status-cure';
  if (S('refresh')) return 'self-status-cure';
  if (S('painsplit')) return 'hp-average (Pain Split)';
  if (S('ingrain') || S('aquaring')) return 'residual-self-heal';
  if (S('batonpass')) return 'volatile-transfer (Baton Pass)';
  if (S('haze')) return 'boost-reset (Haze)';
  if (S('mist')) return 'boost-immunity (Mist)';
  if (S('encore')) return 'move-lock (Encore)';
  if (S('torment')) return 'move-restriction (Torment)';
  if (S('imprison')) return 'move-restriction (Imprison)';
  if (S('meanlook') || S('spiderweb') || S('block')) return 'switch-trap-move';
  if (S('perishsong')) return 'perish-song';
  if (S('destinybond')) return 'reactive-volatile (Destiny Bond)';
  if (S('grudge')) return 'reactive-volatile (Grudge)';
  if (S('endure')) return 'survive-at-1 (Endure)';
  if (S('rapidspin')) return 'hazard-clear (Rapid Spin)';
  if (S('reflect') || S('lightscreen')) return 'screen';
  if (S('safeguard')) return 'side-safeguard';
  if (S('mudsport') || S('watersport')) return 'field-sport';
  if (S('lightscreen')) return 'screen';
  if (S('sunnyday') || S('raindance') || S('sandstorm') || S('hail')) return 'weather-set';
  if (S('bellydrum')) return 'hp-cost-boost (Belly Drum)';
  if (S('curse')) return 'type-conditional (Curse)';
  if (S('defensecurl') || S('minimize')) return 'volatile-self-boost';
  if (S('doubleteam')) return 'evasion-boost';
  if (S('followme')) return 'redirect';
  if (S('camouflage') || S('conversion') || S('conversion2')) return 'type-change';
  if (S('spikes') || S('toxicspikes') || S('stealthrock')) return 'entry-hazard';
  if (S('memento') || S('healingwish') || S('lunardance')) return 'self-faint-effect';
  if (S('trick') || S('switcheroo')) return 'item-swap';
  if (S('psychup')) return 'boost-copy';
  if (S('sketch')) return 'move-copy';
  if (S('spite')) return 'pp-drain';
  if (S('nightmare')) return 'sleep-nightmare';
  if (S('yawn')) return 'delayed-sleep (Yawn)';
  if (S('ingrain')) return 'residual-self-heal';
  if (S('roleplay') || S('skillswap')) return 'ability-swap';
  if (S('recycle')) return 'item-recycle';
  if (S('teeterdance')) return 'confuse';
  if (S('swagger') || S('flatter')) return 'boost+confuse';
  if (S('attract')) return 'attract';
  if (S('charge')) return 'charge-volatile';
  if (S('foresight') || S('odorsleuth') || S('miracleeye')) return 'identify';
  if (S('lockon') || S('mindreader')) return 'lock-on';
  // `confuseray` is MODELED (`gen3_confuse_ray_v1`); its siblings supersonic/sweetkiss are
  // NOT (they share the volatile but not the arm — model them the same way when wanted).
  if (S('supersonic') || S('sweetkiss')) return 'confuse';
  if (S('gravity')) return 'field-gravity';
  if (S('taunt') || S('disable')) return 'move-restriction';
  return 'other-status-move';
}

// ── Load + validate teams (mirroring the e2e gate) ──────────────────────────
function walk(d) {
  let out = [];
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) out = out.concat(walk(p));
    else if (e.name.endsWith('.txt')) out.push(p);
  }
  return out;
}

// ── SCAN_UNIVERSE=1 — classify the ENTIRE gen3-legal move universe, not just the pool.
// The ROUND-40 INVARIANT: after `gen3_unmodeled_move_failloud_v2` the engine has NO
// silent-desync move left in the whole universe — every move is MODELED or FAIL-LOUD, so
// MISMODELED must be EMPTY. Exits non-zero if the invariant is broken (a checkable gate:
// run it after admitting a new move class or touching the guard).
if (process.env.SCAN_UNIVERSE) {
  const buckets = { MODELED: [], MISMODELED: [], UNMODELED: [] };
  for (const m of dex3.moves.all()) {
    if (!m.exists || m.isNonstandard || m.gen > 3 || m.id === 'struggle') continue;
    let cov, mech;
    if (isHiddenPower(m.id)) { cov = 'MODELED'; mech = 'typed-hidden-power'; }
    else if (m.category === 'Status') ({ cov, mech } = classifyStatus(m, m.id));
    else ({ cov, mech } = classifyDamaging(m, m.id));
    buckets[cov].push(`${m.id} (${mech})`);
  }
  const total = buckets.MODELED.length + buckets.MISMODELED.length + buckets.UNMODELED.length;
  console.log(`=== FULL gen3-legal MOVE UNIVERSE (${total} moves) ===`);
  console.log(`MODELED: ${buckets.MODELED.length}  UNMODELED (fail-loud): ${buckets.UNMODELED.length}  MISMODELED (silent desync): ${buckets.MISMODELED.length}`);
  if (buckets.MISMODELED.length) {
    console.log('MISMODELED — the round-40 no-silent-desync invariant is BROKEN:');
    for (const r of buckets.MISMODELED) console.log('  ' + r);
    process.exit(1);
  }
  process.exit(0);
}

const validator = new TeamValidator(VALIDATE_FORMAT);
const files = walk(TEAMS_DIR).sort();

let validTeams = [];       // [{file, moves:Set<id>}]
let importFail = 0, validateFail = 0;

for (const f of files) {
  const raw = fs.readFileSync(f, 'utf8');
  let team;
  try { team = Teams.import(raw); } catch (e) { importFail++; continue; }
  if (!team || !team.length) { importFail++; continue; }
  const problems = validator.validateTeam(team);
  if (problems && problems.length) { validateFail++; continue; }
  const moves = new Set();
  for (const set of team) {
    for (const mv of (set.moves || [])) {
      const id = dex3.moves.get(mv).id; // resolve aliases + normalize
      moves.add(id);
    }
  }
  validTeams.push({ file: path.relative(TEAMS_DIR, f), moves });
}

// ── Per-move coverage classification ─────────────────────────────────────────
const allMoves = new Set();
for (const t of validTeams) for (const id of t.moves) allMoves.add(id);

const moveInfo = {}; // id -> {cov, mech, category, teamCount}
for (const id of allMoves) {
  const m = dex3.moves.get(id);
  let cov, mech;
  if (isHiddenPower(id)) {
    cov = 'MODELED'; mech = 'typed-hidden-power';
  } else if (m.category === 'Status') {
    ({ cov, mech } = classifyStatus(m, id));
  } else {
    ({ cov, mech } = classifyDamaging(m, id));
  }
  moveInfo[id] = { cov, mech, category: m.category, name: m.name, teamCount: 0 };
}
for (const t of validTeams) for (const id of t.moves) moveInfo[id].teamCount++;

// A team is ENGINE-PLAYABLE iff every move it carries is MODELED (MISMODELED counts as
// NOT-clean for the strict "runs bit-for-bit" question, but ALSO reported separately as
// "runs without fail-loud").
function teamBlockers(t, kinds) {
  const bad = [];
  for (const id of t.moves) {
    if (kinds.has(moveInfo[id].cov)) bad.push(id);
  }
  return bad;
}
// Strict: fully MODELED (bit-for-bit). Loose: no FAIL-LOUD (UNMODELED), MISMODELED allowed.
const failLoudKinds = new Set(['UNMODELED']);              // engine panics
const notBitForBit = new Set(['UNMODELED', 'MISMODELED']); // engine wrong or panics

let fullyPlayable = 0, noFailLoud = 0;
for (const t of validTeams) {
  if (teamBlockers(t, notBitForBit).length === 0) fullyPlayable++;
  if (teamBlockers(t, failLoudKinds).length === 0) noFailLoud++;
}

// ── Team-unlock ranking: greedy set cover over the BLOCKING moves ────────────
// A team is unlocked when ALL its blocking moves are modeled. We greedily pick the move
// that unlocks the most ADDITIONAL teams. Two variants: strict (MISMODELED blocks) and
// fail-loud-only.
function greedyCover(blockKinds) {
  // teams that are not yet clean, with their remaining blocker set
  let remaining = validTeams
    .map((t) => ({ file: t.file, blockers: new Set(teamBlockers(t, blockKinds)) }))
    .filter((t) => t.blockers.size > 0);
  const order = [];
  let cumulativeClean = validTeams.length - remaining.length;
  while (remaining.length > 0) {
    // count teams each move would UNLOCK (i.e. teams whose remaining blockers == {move})
    // more precisely: pick the move covering the most teams (appearing in the most
    // remaining blocker-sets); a team is unlocked only when its set becomes empty.
    const freq = {};
    for (const t of remaining) for (const b of t.blockers) freq[b] = (freq[b] || 0) + 1;
    // choose the move in the most teams
    let best = null, bestN = -1;
    for (const [mv, n] of Object.entries(freq)) {
      if (n > bestN || (n === bestN && mv < best)) { best = mv; bestN = n; }
    }
    // remove best from all remaining; teams that become empty are unlocked
    let unlockedNow = 0;
    for (const t of remaining) {
      if (t.blockers.delete(best) && t.blockers.size === 0) unlockedNow++;
    }
    remaining = remaining.filter((t) => t.blockers.size > 0);
    cumulativeClean += unlockedNow;
    order.push({
      move: best, name: moveInfo[best].name, mech: moveInfo[best].mech,
      cov: moveInfo[best].cov, teamsCarrying: moveInfo[best].teamCount,
      teamsInBlockingSet: bestN, teamsUnlockedNow: unlockedNow, cumulativeClean,
    });
  }
  return order;
}

const strictCover = greedyCover(notBitForBit);
const failLoudCover = greedyCover(failLoudKinds);

// ── Ranking by raw team-carry (unmodeled + mismodeled) ───────────────────────
const unmodeledRanked = Object.entries(moveInfo)
  .filter(([, v]) => v.cov !== 'MODELED')
  .sort((a, b) => b[1].teamCount - a[1].teamCount)
  .map(([id, v]) => ({ move: id, name: v.name, cov: v.cov, mech: v.mech, category: v.category, teamCount: v.teamCount }));

const out = {
  totals: {
    txtFiles: files.length,
    importFail, validateFail,
    validTeams: validTeams.length,
    distinctMoves: allMoves.size,
    distinctModeled: Object.values(moveInfo).filter((v) => v.cov === 'MODELED').length,
    distinctUnmodeledFailLoud: Object.values(moveInfo).filter((v) => v.cov === 'UNMODELED').length,
    distinctMismodeled: Object.values(moveInfo).filter((v) => v.cov === 'MISMODELED').length,
    fullyPlayableTeams: fullyPlayable,   // every move MODELED bit-for-bit
    noFailLoudTeams: noFailLoud,          // no UNMODELED (MISMODELED allowed — runs, maybe wrong)
  },
  unmodeledRanked,
  strictCover,
  failLoudCover,
};

if (process.env.SCAN_JSON) {
  process.stdout.write(JSON.stringify(out, null, 2));
} else {
  const T = out.totals;
  console.log('=== ENGINE MOVE-COVERAGE SCAN (gen3ou team pool) ===');
  console.log(`.txt files: ${T.txtFiles}  import-fail: ${T.importFail}  validate-fail: ${T.validateFail}  VALID: ${T.validTeams}`);
  console.log(`distinct moves across valid teams: ${T.distinctMoves}`);
  console.log(`  MODELED (engine runs bit-for-bit): ${T.distinctModeled}`);
  console.log(`  MISMODELED (engine RUNS, sub-mechanic wrong — silent desync): ${T.distinctMismodeled}`);
  console.log(`  UNMODELED (engine FAIL-LOUDs / panics): ${T.distinctUnmodeledFailLoud}`);
  console.log('');
  console.log(`teams FULLY engine-playable (all moves MODELED bit-for-bit): ${T.fullyPlayableTeams} / ${T.validTeams}`);
  console.log(`teams with NO FAIL-LOUD (MISMODELED allowed): ${T.noFailLoudTeams} / ${T.validTeams}`);
  console.log('');
  console.log('=== RANKED unmodeled/mismodeled moves by # teams carrying ===');
  console.log('cov          #teams  cat       mech                              move');
  for (const r of out.unmodeledRanked) {
    console.log(`${r.cov.padEnd(11)} ${String(r.teamCount).padStart(5)}   ${r.category.padEnd(9)} ${r.mech.padEnd(33)} ${r.move}`);
  }
  console.log('');
  console.log('=== GREEDY SET-COVER (STRICT: MISMODELED counts as a blocker) ===');
  console.log('the sequence of moves/classes to model → cumulative teams bit-for-bit playable');
  console.log('rank  move                 cov         mech                         carrying  unlocks  cumClean');
  out.strictCover.forEach((s, i) => {
    console.log(`${String(i + 1).padStart(3)}   ${s.move.padEnd(20)} ${s.cov.padEnd(11)} ${s.mech.padEnd(28)} ${String(s.teamsCarrying).padStart(6)}  ${String(s.teamsUnlockedNow).padStart(6)}  ${String(s.cumulativeClean).padStart(6)}`);
  });
  console.log('');
  console.log('=== GREEDY SET-COVER (FAIL-LOUD only: MISMODELED allowed to run) ===');
  console.log('rank  move                 mech                         carrying  unlocks  cumClean');
  out.failLoudCover.forEach((s, i) => {
    console.log(`${String(i + 1).padStart(3)}   ${s.move.padEnd(20)} ${s.mech.padEnd(28)} ${String(s.teamsCarrying).padStart(6)}  ${String(s.teamsUnlockedNow).padStart(6)}  ${String(s.cumulativeClean).padStart(6)}`);
  });
}
