// probe_facade_gen3.js — settle the gen3 FACADE model bit-for-bit (the A/B fuzzer's
// #1 open cluster). The RESOLVED `Dex.forFormat('gen3customgame')` sim is the ONLY oracle.
//
// Questions (each answered by direct measurement, not source reading alone):
//  (a) WHICH statuses trigger the boost — brn / par / psn / tox? (slp excluded per the
//      dist source guard; frz unreachable-while-acting since Facade has no defrost flag)
//  (b) WHERE it folds — the move's own `onBasePower` returning `chainModify(2)` = a
//      BP-chain member (one accumulated 4096 modifier, like the pinch family), or a
//      direct multiply? Discriminator: compose with a Pink Bow (a DIRECT float ×1.1 that
//      makes relayVar non-integer → the runEvent tail SKIPS the accumulated chain).
//      chain-skipped ⇒ BP 77 (bow only); both-fold ⇒ BP 154.
//  (c) Does the gen3 BURN damage-halve still apply to a burned Facade? (hypothesis YES —
//      gen3 Facade does NOT ignore burn; measure brn vs psn damage.)
//  (d) GUTS composition — a burned Guts Facade user: Atk ×1.5 + burn-halve SUPPRESSED +
//      BP ×2?
//
// Method: direct in-process Battle (gen3customgame — no clause shuffles), attacker
// status FORCE-SET via pokemon.setStatus() before the measured turn (draw-free,
// outside the count), ONE Facade turn measured. Max-roll damage = max defender HP
// delta over a seed sweep, crit turns EXCLUDED by scanning the turn log for |-crit|
// (the gen_damage_golden discipline). Full-para aborts (delta 0) drop out of the max.
//
// Run:  node src/rust_sim/harness/probe_facade_gen3.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// One measured turn: p1 uses `moveId` into p2 (who splashes). Returns the defender HP
// delta, whether the turn logged a crit, whether p1 actually moved, and p1's status
// before/after.
function runOneTurn({ seed, attacker, defender, moveId, status, toxStage }) {
  const battle = new Battle({ formatid: 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([attacker]) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack([defender]) });
  const atk = battle.sides[0].active[0];
  const def = battle.sides[1].active[0];
  if (status) {
    atk.setStatus(status);
    if (status === 'tox' && toxStage) atk.statusState.stage = toxStage;
  }
  const statusBefore = atk.status || 'none';
  const hpBefore = def.hp;
  const logStart = battle.log.length;
  battle.choose('p1', 'move ' + moveId);
  battle.choose('p2', 'move splash');
  const turnLog = battle.log.slice(logStart).join('\n');
  return {
    delta: hpBefore - def.hp,
    crit: turnLog.includes('|-crit|'),
    moved: turnLog.includes(`|move|p1a:`),
    statusBefore,
    statusAfter: atk.status || 'none',
  };
}

// Max-roll (crit-excluded) damage over a seed sweep. Also report the count of distinct
// non-crit deltas seen (sanity: 16-roll spread) and the second-highest (the r==1 check).
function maxRoll(cfg, nSeeds = 300) {
  const deltas = new Set();
  for (let s = 1; s <= nSeeds; s++) {
    const r = runOneTurn({ ...cfg, seed: [0, 0, 0, s] });
    if (!r.crit && r.moved && r.delta > 0) deltas.add(r.delta);
  }
  const sorted = [...deltas].sort((a, b) => b - a);
  return { max: sorted[0], second: sorted[1], nDistinct: sorted.length };
}

function main() {
  const dex3 = Dex.forFormat('gen3customgame');
  const facade = dex3.moves.get('facade');
  console.log('=== resolved gen3 facade ===');
  console.log('  basePower =', facade.basePower, ' type =', facade.type, ' category =', facade.category);
  console.log('  flags =', JSON.stringify(facade.flags));
  console.log('  onBasePower =', facade.onBasePower ? facade.onBasePower.toString().replace(/\s+/g, ' ') : 'NONE');
  console.log('  basePowerCallback =', typeof facade.basePowerCallback);

  // Attacker: Raticate (the randbats facade carrier), neutral ability by default.
  // Defender: Snorlax, no item, No Ability, splash-only.
  const A = (opts = {}) => mon('Raticate', ['facade'], opts);
  const D = mon('Snorlax', ['splash']);

  console.log('\n=== (a) status trigger sweep — max-roll (crit-excluded) Facade damage ===');
  const arms = [
    ['none', null],
    ['psn', 'psn'],
    ['tox', 'tox'],
    ['par', 'par'],
    ['brn', 'brn'],
  ];
  const results = {};
  for (const [label, st] of arms) {
    const r = maxRoll({ attacker: A(), defender: D, moveId: 'facade', status: st });
    results[label] = r.max;
    console.log(`  status=${label.padEnd(4)} maxRoll=${r.max}  (2nd=${r.second}, distinct=${r.nDistinct})`);
  }

  console.log('\n=== (c) burn-halve check ===');
  console.log(`  brn=${results.brn} vs psn=${results.psn} → burn ${results.brn < results.psn ? 'STILL HALVES (≈half)' : 'does NOT halve'}`);

  console.log('\n=== (d) Guts composition (burned Guts Facade) ===');
  const guts = maxRoll({ attacker: A({ ability: 'Guts' }), defender: D, moveId: 'facade', status: 'brn' });
  console.log(`  guts+brn maxRoll=${guts.max}  (psn no-guts=${results.psn} → expect ×1.5 of psn if Atk×1.5 + halve-suppressed + BP×2)`);
  const gutsNoStatus = maxRoll({ attacker: A({ ability: 'Guts' }), defender: D, moveId: 'facade', status: null });
  console.log(`  guts no-status maxRoll=${gutsNoStatus.max} (control: Guts inert unstatused → expect == none=${results.none})`);

  console.log('\n=== (b) fold discriminator — Pink Bow (DIRECT ×1.1 float) + poisoned Facade ===');
  const bowPsn = maxRoll({ attacker: A({ item: 'Pink Bow' }), defender: D, moveId: 'facade', status: 'psn' });
  const bowNone = maxRoll({ attacker: A({ item: 'Pink Bow' }), defender: D, moveId: 'facade', status: null });
  console.log(`  pinkbow+psn maxRoll=${bowPsn.max}  pinkbow no-status maxRoll=${bowNone.max}  plain none=${results.none}, plain psn=${results.psn}`);
  console.log('  (if the runEvent integer-guard SKIPS the facade chain after the bow float: bow+psn ≈ bow-none;');
  console.log('   if both fold: bow+psn ≈ psn ×1.1)');

  console.log('\n=== control: Body Slam (no onBasePower) is NOT status-boosted ===');
  const bsNone = maxRoll({ attacker: mon('Raticate', ['bodyslam']), defender: D, moveId: 'bodyslam', status: null });
  const bsPsn = maxRoll({ attacker: mon('Raticate', ['bodyslam']), defender: D, moveId: 'bodyslam', status: 'psn' });
  console.log(`  bodyslam none=${bsNone.max} psn=${bsPsn.max} (must be equal)`);

  console.log('\n=== draw-count check: does the facade boost consume PRNG? ===');
  // Count raw draws on a psn facade turn vs a none facade turn at the same seed.
  for (const st of [null, 'psn']) {
    const battle = new Battle({ formatid: 'gen3customgame', seed: [0, 0, 0, 42] });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack([A()]) });
    battle.setPlayer('p2', { name: 'B', team: Teams.pack([D]) });
    if (st) battle.sides[0].active[0].setStatus(st);
    let draws = 0;
    const backend = battle.prng.rng;
    const orig = backend.next.bind(backend);
    backend.next = (...a) => { draws++; return orig(...a); };
    battle.choose('p1', 'move facade');
    battle.choose('p2', 'move splash');
    console.log(`  status=${st || 'none'} turn draws=${draws}`);
  }
}

main();
