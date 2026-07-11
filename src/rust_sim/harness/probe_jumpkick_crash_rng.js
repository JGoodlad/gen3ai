// probe_jumpkick_crash_rng.js — settle the gen3 JUMP KICK / HIGH JUMP KICK crash-damage
// model bit-for-bit (the handler-completeness audit's first real miss).
//
// THE GAP: `isModeledMove` admits jumpkick/highjumpkick (plain damaging moves, no
// secondary), but the resolved gen3 dist carries an `onMoveFail` the port never
// enumerated:
//   onMoveFail(target, source, move) {
//     if (target.runImmunity("Fighting")) {
//       const damage = this.actions.getDamage(source, target, move, true);
//       this.damage(this.clampIntRange(damage / 2, 1, Math.floor(target.maxhp / 2)),
//                   source, source, move);
//     }
//   }
// A missed JK/HJK crashes the USER in the sim while the port does nothing — a silent
// HP divergence. This probe pins:
//   A. the DRAW MODEL of a missed JK — does the crash's `getDamage(..., true)` draw the
//      crit roll and/or the 16-way damage roll? (compare draw counts: miss vs hit vs a
//      missed control move with no onMoveFail)
//   B. the exact crash VALUE (per seed) so the port's mirror can be asserted
//   C. a miss vs a GHOST (runImmunity false) — no crash, and the draw count
//   D. JK into a PROTECTING target — does onMoveFail fire (crash through Protect)?
//   E. JK MISSING a target behind a SUBSTITUTE — crash still fires? and a JK ABSORBED
//      by the sub (a hit) — no crash
//   F. crash lethality — the user can crash itself to 0 (faint ordering)
//
// Run:  node src/rust_sim/harness/probe_jumpkick_crash_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// One JK turn on a fresh battle at `seed`. Returns per-turn draw count, the user's
// HP loss, the target's HP loss, and the raw log tail.
function runTurn({ seed, p1moves, p2moves, p1choice, p2choice, p1mon, p2mon, preTurns }) {
  const battle = new Battle({ formatid: 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([p1mon || mon('Hitmonlee', p1moves)]) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack([p2mon || mon('Snorlax', p2moves, { evs: { hp: 252 } })]) });
  for (const [c1, c2] of preTurns || []) {
    battle.choose('p1', c1);
    battle.choose('p2', c2);
  }
  const user = battle.sides[0].active[0];
  const tgt = battle.sides[1].active[0];
  const hpU0 = user.hp, hpT0 = tgt.hp;
  let draws = 0;
  const backend = battle.prng.rng;
  const origNext = backend.next.bind(backend);
  backend.next = (...a) => { draws++; return origNext(...a); };
  const logStart = battle.log.length;
  battle.choose('p1', p1choice || 'move jumpkick');
  battle.choose('p2', p2choice || 'move splash');
  backend.next = origNext;
  const log = battle.log.slice(logStart).filter((l) => /\|-damage|\|move\||-miss|-immune|-fail|faint|-activate|-start/.test(l));
  return {
    draws,
    userLoss: hpU0 - user.hp, tgtLoss: hpT0 - tgt.hp,
    userHp: user.hp, userMax: user.maxhp, tgtMax: tgt.maxhp,
    missed: battle.log.slice(logStart).some((l) => l.includes('|-miss|')),
    log,
  };
}

function seedsWhere(pred, base, n, mk) {
  const out = [];
  for (let s = 0; s < 4000 && out.length < n; s++) {
    const seed = [base + s * 7 + 1, 2 * s + 3, 5 * s + 11, s + 17].map((x) => (x % 65536) || 1);
    const r = mk(seed);
    if (pred(r)) out.push({ seed, r });
  }
  return out;
}

function main() {
  console.log('=== A/B: draw counts + crash value, JK hit vs miss (Hitmonlee JK vs Snorlax) ===');
  const mk = (seed) => runTurn({ seed, p1moves: ['jumpkick'], p2moves: ['splash'] });
  const hits = seedsWhere((r) => !r.missed, 100, 3, mk);
  const misses = seedsWhere((r) => r.missed, 100, 6, mk);
  for (const { seed, r } of hits) {
    console.log(`  HIT  seed=${seed}  draws=${r.draws}  userLoss=${r.userLoss}  tgtLoss=${r.tgtLoss}`);
  }
  for (const { seed, r } of misses) {
    console.log(`  MISS seed=${seed}  draws=${r.draws}  userLoss=${r.userLoss} (crash)  tgtLoss=${r.tgtLoss}  maxhp/2=${Math.floor(r.userMax / 2)}`);
    console.log(`       log: ${r.log.join('  ')}`);
  }

  console.log('=== control: missed AERIAL ACE-like (no onMoveFail) — use Karate Chop misses ===');
  const mkc = (seed) => runTurn({ seed, p1moves: ['karatechop'], p2moves: ['splash'], p1choice: 'move karatechop' });
  const cmiss = seedsWhere((r) => r.missed, 100, 2, mkc);
  for (const { seed, r } of cmiss) console.log(`  MISS(control) seed=${seed} draws=${r.draws} userLoss=${r.userLoss}`);
  console.log('  (karatechop acc100 never misses — expect none; fall back to Rock Slide acc90)');
  const mkr = (seed) => runTurn({ seed, p1moves: ['rockslide'], p2moves: ['splash'], p1choice: 'move rockslide' });
  for (const { seed, r } of seedsWhere((r) => r.missed, 100, 3, mkr)) {
    console.log(`  MISS(rockslide) seed=${seed} draws=${r.draws} userLoss=${r.userLoss}`);
  }

  console.log('=== C: JK vs GHOST (Gengar) — immune: crash? draws? ===');
  const mkg = (seed) => runTurn({ seed, p1moves: ['jumpkick'], p2mon: mon('Gengar', ['splash']), p2moves: ['splash'] });
  for (let s = 0; s < 3; s++) {
    const seed = [s * 13 + 21, s + 5, 3 * s + 9, 41].map((x) => x || 1);
    const r = mkg(seed);
    console.log(`  GHOST seed=${seed} draws=${r.draws} userLoss=${r.userLoss} tgtLoss=${r.tgtLoss} log: ${r.log.join('  ')}`);
  }

  console.log('=== D: JK into PROTECT — crash through the block? ===');
  const mkp = (seed) => runTurn({
    seed, p1moves: ['jumpkick'], p2mon: mon('Snorlax', ['protect'], { evs: { hp: 252 } }),
    p2choice: 'move protect',
  });
  for (let s = 0; s < 4; s++) {
    const seed = [s * 29 + 7, s * 3 + 1, 11, s + 51].map((x) => x || 1);
    const r = mkp(seed);
    console.log(`  PROTECT seed=${seed} draws=${r.draws} missed=${r.missed} userLoss=${r.userLoss} log: ${r.log.join('  ')}`);
  }

  console.log('=== E: JK vs a SUB — absorbed hit (no crash) vs miss (crash) ===');
  const mks = (seed) => runTurn({
    seed, p1moves: ['jumpkick'], p2mon: mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }),
    preTurns: [['move jumpkick', 'move substitute']], p2choice: 'move splash',
  });
  const subHit = seedsWhere((r) => !r.missed, 300, 2, mks);
  const subMiss = seedsWhere((r) => r.missed, 300, 2, mks);
  for (const { seed, r } of subHit) console.log(`  SUB-HIT  seed=${seed} draws=${r.draws} userLoss=${r.userLoss} log: ${r.log.join('  ')}`);
  for (const { seed, r } of subMiss) console.log(`  SUB-MISS seed=${seed} draws=${r.draws} userLoss=${r.userLoss} log: ${r.log.join('  ')}`);

  console.log('=== F: crash can faint the user (low-HP Hitmonlee misses) ===');
  // Pre-damage the user via a couple of Seismic Tosses, then find a miss.
  const mkf = (seed) => runTurn({
    seed, p1moves: ['jumpkick'], p2mon: mon('Snorlax', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    preTurns: [['move jumpkick', 'move seismictoss'], ['move jumpkick', 'move seismictoss']],
    p2choice: 'move splash',
  });
  for (const { seed, r } of seedsWhere((r) => r.missed, 900, 2, mkf)) {
    console.log(`  LOWHP-MISS seed=${seed} draws=${r.draws} userLoss=${r.userLoss} userHp=${r.userHp} log: ${r.log.join('  ')}`);
  }

  console.log('=== HJK sanity (same shape, acc 90) ===');
  const mkh = (seed) => runTurn({ seed, p1moves: ['highjumpkick'], p1choice: 'move highjumpkick', p2moves: ['splash'] });
  for (const { seed, r } of seedsWhere((r) => r.missed, 100, 2, mkh)) {
    console.log(`  HJK-MISS seed=${seed} draws=${r.draws} userLoss=${r.userLoss} log: ${r.log.join('  ')}`);
  }
}

main();
