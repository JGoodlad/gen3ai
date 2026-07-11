// probe_taunt_duration_branch.js — nail gen3 Taunt's stored duration for the two onStart
// branches, vs the omniscient sim. gen3 taunt: `duration: 2`, onStart INHERITS the base's
// `if (target.activeTurns && !willMove(target)) duration++`. So:
//   - taunter FASTER (target still to move → willMove TRUE): duration stays 2.
//   - taunter SLOWER on turn 1 (target.activeTurns == 0): the activeTurns gate is FALSE → 2.
//   - taunter SLOWER on turn >=2 (target.activeTurns >= 1 AND willMove FALSE): duration++ → 3.
// We hook addVolatile(taunt) to read activeTurns + willMove + the post-onStart duration, and
// the residual to show the tick-down + free-up. Ground truth for MINOR A.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { Pokemon } = require(path.join(PS, 'dist/sim/pokemon'));

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

async function run(label, p1team, p2team, plan, tauntTurnIdx, targetSide) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[7,11,13,17]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const tgt = () => battle.sides[targetSide].active[0];

  const events = [];
  const realAdd = Pokemon.prototype.addVolatile;
  Pokemon.prototype.addVolatile = function (status, source, sourceEffect, linkedStatus) {
    const id = (typeof status === 'string') ? this.battle.dex.conditions.get(status).id : status.id;
    if (id === 'taunt') {
      const activeTurns = this.activeTurns;
      const willMove = !!this.battle.queue.willMove(this);
      const ret = realAdd.call(this, status, source, sourceEffect, linkedStatus);
      const post = this.volatiles.taunt ? this.volatiles.taunt.duration : null;
      events.push(`  addVolatile(taunt): target.activeTurns=${activeTurns}  willMove(target)=${willMove}  → post-onStart duration=${post}`);
      return ret;
    }
    return realAdd.call(this, status, source, sourceEffect, linkedStatus);
  };
  const realFieldEvent = battle.fieldEvent.bind(battle);
  battle.fieldEvent = function (eventid, targets) {
    if (eventid === 'Residual') {
      const b = tgt() && tgt().volatiles.taunt ? tgt().volatiles.taunt.duration : null;
      const r = realFieldEvent(eventid, targets);
      const a = tgt() && tgt().volatiles.taunt ? tgt().volatiles.taunt.duration : null;
      if (b !== null || a !== null) events.push(`  [turn ${battle.turn}] residual: taunt ${b} → ${a}${a === null ? '  (FREED)' : ''}`);
      return r;
    }
    return realFieldEvent(eventid, targets);
  };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 14) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    if (i > tauntTurnIdx + 4 && !(tgt() && tgt().volatiles.taunt)) break;
  }
  Pokemon.prototype.addVolatile = realAdd;
  try { streams.omniscient.destroy(); } catch (e) {}
  console.log(`\n==== ${label} ====`);
  console.log(events.join('\n'));
}

async function main() {
  // A: taunter FASTER (Alakazam) taunts on turn 1 → activeTurns==0, willMove TRUE → duration 2.
  await run('A: FASTER taunter, turn 1 (activeTurns 0, willMove TRUE)',
    [mon('Alakazam', ['taunt', 'psychic'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'toxic'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], 0, 1);

  // B: taunter SLOWER (Snorlax) taunts a FASTER Alakazam on turn 2+ → activeTurns>=1,
  //    willMove FALSE → duration++ → 3. Turn 1 both attack (so target has activeTurns), turn 2
  //    Snorlax Taunts (slower, Alakazam moved first).
  await run('B: SLOWER taunter, turn 2 (target activeTurns>=1, willMove FALSE → duration++)',
    [mon('Snorlax', ['taunt', 'bodyslam'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } })],
    [mon('Alakazam', ['psychic', 'calmmind'], { nature: 'Timid', evs: { hp: 252, spa: 252, spe: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }], 1, 1);

  // C: taunter SLOWER but on TURN 1 (target.activeTurns==0) → gate FALSE → duration stays 2
  //    (even though willMove FALSE). Snorlax Taunts a faster Alakazam on turn 1.
  await run('C: SLOWER taunter, turn 1 (target activeTurns 0 → NO duration++)',
    [mon('Snorlax', ['taunt', 'bodyslam'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } })],
    [mon('Alakazam', ['psychic', 'calmmind'], { nature: 'Timid', evs: { hp: 252, spa: 252, spe: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }], 0, 1);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
