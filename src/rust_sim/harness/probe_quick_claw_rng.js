// probe_quick_claw_rng.js — GROUND-TRUTH for the gen3 Quick Claw speed=65535 override
// (`gen3_quick_claw_speed_v1`, the P1/P2 fix). A SLOW Quick-Claw holder whose end-of-
// PREV-turn `battle.quickClawRoll` (randomChance(1,5)) hit TRUE moves FIRST next turn
// (gen3 getActionSpeed → speed=65535), regardless of raw Speed. We construct a clearly-
// distinct-speed pair (fast p1 vs slow QC p2) and search a start seed where turn-1's
// endTurn roll hits TRUE, then report per-turn firstMover + seedAfter + HP so the Rust
// regression test can pin the exact state.
'use strict';
const path = require('path');
const PS = path.join(__dirname, '..', '..', '..', 'deps', 'pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const initSeed = String(battle.prng.getSeed()); // post-construction (No-Ability mons draw nothing)
  const rows = [];
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const logStart = log.length;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    // First |move| line this turn -> first mover side.
    let firstMover = '?';
    for (let j = logStart; j < log.length; j++) {
      const m = log[j].match(/^\|move\|(p[12])/);
      if (m) { firstMover = m[1]; break; }
    }
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    rows.push({ turn: i, firstMover, seedAfter: String(after),
      p1: a0 ? `${a0.hp}/${a0.maxhp}` : '-', p2: a1 ? `${a1.hp}/${a1.maxhp}` : '-',
      qcRoll: battle.quickClawRoll });
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { initSeed, rows };
}

async function main() {
  // p1 Electrode: FAST (base spe 150), no item. p2 Shuckle: SLOW (base spe 5) + Quick Claw.
  // Both use Swift (never-miss, no secondary) so the turn draws are minimal + no faint.
  const p1 = [mon('Electrode', ['swift'], { nature: 'Serious' })];
  const p2 = [mon('Shuckle', ['swift'], { item: 'Quick Claw', nature: 'Serious' })];
  const plan = [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' },
                { p1: 'move 1', p2: 'move 1' }];
  // Search seeds for one where turn-1 endTurn quickClawRoll=TRUE -> turn-2 firstMover=p2.
  for (let s = 0; s < 60; s++) {
    const seed = [s, s * 7 + 1, s * 13 + 3, s * 17 + 5];
    const { initSeed, rows } = await run(seed, p1, p2, plan);
    if (rows.length >= 2 && rows[0].qcRoll === true && rows[1].firstMover === 'p2') {
      console.log(`FOUND seed=${JSON.stringify(seed)} initSeed=${initSeed}`);
      for (const r of rows) console.log('  ', JSON.stringify(r));
      return;
    }
  }
  console.log('no reordering seed found in search range');
}
main();
