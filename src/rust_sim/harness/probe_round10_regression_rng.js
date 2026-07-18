// probe_round10_regression_rng.js — GROUND-TRUTH for the ROUND-10 fixes' regression pins
// (tests/regression_test.rs). Unlike the residual-INJECT probes, these PLAY scripted turns
// through the omniscient sim (mirroring the Rust `run_full_battle`), print the per-decision
// post-decision PRNG seed + relevant state/lines, and (BR1) dump the last-mon recharge
// `|request|` JSON. Each captures the sim's PRE-first-decision seed (post-construction) — the
// RAW seed the Rust regression test feeds `start_with_switchins`.
//
//   RB1 — encore + choicelock co-present at endTurn draws the size-2 DisableMove tie-shuffle.
//   RM1 — Brick Break removes Reflect + deals full (non-halved) damage (STATE + both-screens seed).
//   RM3 — the sand `|-weather|…[upkeep]` line under Air Lock precedes the leech `|-damage|`,
//         and the constructed scenario is DRAW-NEUTRAL (post-turn seed vs a control).
//   BR1 — a move-LOCKED LAST mon (recharge / two-turn fire) request carries `trapped:true`.
//
// Run:  node src/rust_sim/harness/probe_round10_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Play `plans` (an array of {p1,p2} `move K`/`switch N` strings), one per decision boundary,
// through the sim. Returns { initSeed, perDecision:[{seedAfter}], lines, request(last) }.
async function play(label, startSeed, p1team, p2team, plans, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  const p1reqs = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) lines.push(l); })();
  (async () => {
    for await (const ch of streams.p1) {
      for (const l of ch.split('\n')) { if (l.startsWith('|request|')) p1reqs.push(l.slice('|request|'.length)); }
    }
  })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(startSeed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const initSeed = battle.prng.getSeed();
  const perDecision = [];
  for (const plan of plans) {
    if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
    if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    perDecision.push({ seedAfter: battle.prng.getSeed() });
  }
  const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''}` : '-';
  console.log(`\n=== ${label} ===  startSeed=${startSeed.join(',')}`);
  console.log(`  initSeed(RAW for Rust)=${initSeed}`);
  perDecision.forEach((d, i) => console.log(`  dec${i} seedAfter=${d.seedAfter}`));
  console.log(`  p1=${fmt(a0)}  reflect=${battle.sides[0].sideConditions.reflect ? 'UP' : '-'} ls=${battle.sides[0].sideConditions.lightscreen ? 'UP' : '-'}`);
  console.log(`  p2=${fmt(a1)}  reflect=${battle.sides[1].sideConditions.reflect ? 'UP' : '-'} ls=${battle.sides[1].sideConditions.lightscreen ? 'UP' : '-'}`);
  if (opts.showLines) {
    console.log('  --- filtered lines (last turn window) ---');
    for (const l of lines) if (/-weather|Leech Seed|-damage|-sideend|upkeep|move\|/.test(l)) console.log('   ', l);
  }
  if (opts.showReq) {
    console.log('  --- p1 |request| JSONs ---');
    p1reqs.forEach((r, i) => console.log(`    req${i}: ${r}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { initSeed, perDecision, lines, p1reqs };
}

async function main() {
  // ---------------- RB1 ----------------
  // A Choice-Band mon (choicelock) whose FOE Encores it → at that turn's endTurn the mon
  // carries BOTH choicelock + encore → runEvent('DisableMove') draws one size-2 tie-shuffle
  // BEFORE the Quick Claw. Snorlax (Choice Band, FAST) Body Slams (sets choicelock+lastMove);
  // p2 BULKY Blissey (survives the Body Slam so the turn REACHES endTurn) Encores the Snorlax.
  // Control: choicelock-only (Blissey Splashes) — n stays 1, NO shuffle → the seed DIFFERS,
  // proving the extra draw the fix adds.
  await play('RB1: encore+choicelock endTurn shuffle',
    [11, 22, 33, 44],
    [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'choiceband', ability: 'Thick Fat', nature: 'Jolly', evs: { atk: 252, spd: 4, spe: 252 } })],
    [mon('Blissey', ['encore', 'splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252, spe: 0 } })],
    // Turn 1: Snorlax Body Slam (locks) ; Blissey Encore the Snorlax. endTurn: shuffle.
    [{ p1: 'move 1', p2: 'move 1' }]);
  await play('RB1 CONTROL: choicelock-only (no encore) → NO shuffle',
    [11, 22, 33, 44],
    [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'choiceband', ability: 'Thick Fat', nature: 'Jolly', evs: { atk: 252, spd: 4, spe: 252 } })],
    [mon('Blissey', ['splash', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252, spe: 0 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ---------------- RM1 ----------------
  // p1 Machamp Brick Break; p2 Snorlax uses Reflect turn 1, then p1 Brick Breaks turn 2 → the
  // screen is removed + full damage. Control A: NO reflect (baseline full damage). Control B:
  // BOTH screens up → Brick Break removes both, no ModifyDamagePhase1 tie-shuffle (seed ==
  // one-screen control).
  await play('RM1: Brick Break into Reflect (removes + full dmg)',
    [7, 15, 23, 31],
    [mon('Machamp', ['brickbreak', 'splash'], { nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    [mon('Snorlax', ['reflect', 'splash'], { ability: 'Thick Fat', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    // Turn 1: p1 Splash, p2 Reflect. Turn 2: p1 Brick Break, p2 Splash.
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }], { showLines: true });
  await play('RM1 CONTROL A: Brick Break into NO screen (baseline dmg)',
    [7, 15, 23, 31],
    [mon('Machamp', ['brickbreak', 'splash'], { nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    [mon('Snorlax', ['reflect', 'splash'], { ability: 'Thick Fat', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 2' }]);

  // ---------------- RM3 ----------------
  // p1 Tyranitar (Sand Stream) + Leech Seed vs p2 Golduck (Cloud Nine) leech-seeded — the
  // residual must emit `|-weather|Sandstorm|[upkeep]` (order 8) BEFORE the leech `|-damage|`
  // (order 10.5), and be DRAW-NEUTRAL. Turn 1: p1 Leech Seed, p2 Splash. Turn 2 (record): both
  // Splash → the residual. Golduck is Water (leech lands), grounded, non-Rock/Ground/Steel.
  await play('RM3: sand upkeep under Cloud Nine before leech damage',
    [3, 9, 27, 81],
    [mon('Tyranitar', ['leechseed', 'splash'], { ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    [mon('Golduck', ['splash'], { ability: 'Cloud Nine', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], { showLines: true });

  // ---------------- BR1 ----------------
  // A last-mon move-LOCKED recharge turn: p1 is a SINGLE-mon team (Snorlax w/ Hyper Beam), so
  // after Hyper Beam hits, the recharge turn is a no-bench state. The request must carry
  // `trapped:true`. p2 a bulky wall (Steelix) so Snorlax survives to recharge.
  await play('BR1: last-mon recharge request trapped',
    [5, 5, 5, 5],
    [mon('Snorlax', ['hyperbeam', 'splash'], { nature: 'Brave', evs: { hp: 252, atk: 252 }, level: 5 })],
    [mon('Steelix', ['irondefense', 'splash'], { ability: 'Sturdy', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    // T1: Snorlax Hyper Beam (hits Steelix, then mustrecharge). T2: recharge turn (record req).
    [{ p1: 'move 1', p2: 'move 2' }, { p1: 'move 1', p2: 'move 2' }], { showReq: true });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
