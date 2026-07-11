// probe_statusimmune_setstatus_event.js — the SURGICAL SetStatus-event measurement.
//
// Wraps `battle.runEvent` to trap EVERY `SetStatus` and `Immunity` invocation and, INSIDE
// each, (a) records the number of handlers the sim GATHERED (via findEventHandlers, the
// tie-shuffle size determinant) and (b) counts the PRNG draws consumed WITHIN that event.
// It also records whether the event fired at all (onImmunity blocks at runStatusImmunity,
// a DIFFERENT event from SetStatus).
//
// This isolates THE CRUX with zero downstream confound:
//   - onSetStatus member -> the SetStatus event is REACHED; handler count = 2(clauses)+1 in
//     gen3ou (size-3 shuffle = 1 draw), 0+1=1 in customgame (NO shuffle, 0 draws).
//   - onImmunity member (Magma Armor) -> the block is at the `Immunity` event; the SetStatus
//     event is NOT reached (or reached with only the clause handlers). So it does NOT change
//     the SetStatus handler count.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

async function probe(fmt, ability, mv, species, seed, isSecondary) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":${JSON.stringify(seed)}}`);
  const attackerMove = isSecondary ? 'icebeam' : mv;
  const attackerSpecies = isSecondary ? 'Articuno' : 'Blissey';
  const p1team = [mon(attackerSpecies, [attackerMove, 'recover'])];
  const p2team = [mon(species, ['rest', 'bodyslam'], { ability })];
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  const events = [];
  let curEvent = null;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { if (curEvent) curEvent.draws += 1; return realNext(...a); };

  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, target, source, sourceEffect, relayVar, onEffect, fastExit) {
    if (eventid === 'SetStatus' || eventid === 'Immunity') {
      // Peek the gathered handler count (findEventHandlers is what runEvent gathers).
      let nHandlers = -1;
      try {
        const handlers = battle.findEventHandlers(target, eventid, source);
        nHandlers = handlers.length;
      } catch (e) { nHandlers = `err:${e.message}`; }
      const prev = curEvent;
      curEvent = { eventid, nHandlers, draws: 0, relayVar };
      const ret = realRun(eventid, target, source, sourceEffect, relayVar, onEffect, fastExit);
      curEvent.ret = ret;
      events.push(curEvent);
      curEvent = prev;
      return ret;
    }
    return realRun(eventid, target, source, sourceEffect, relayVar, onEffect, fastExit);
  };

  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 12; k++) await tick();
  const tgt = battle.sides[1].active[0];
  return { events, tgtStatus: tgt.status || '' };
}

const MEMBERS = [
  ['Limber',       'thunderwave', 'Snorlax', false],
  ['Insomnia',     'spore',       'Snorlax', false],
  ['Vital Spirit', 'spore',       'Snorlax', false],
  ['Immunity',     'toxic',       'Snorlax', false],
  ['Water Veil',   'willowisp',   'Snorlax', false],
  ['Magma Armor',  'icebeam',     'Snorlax', true],   // frz secondary
  // Controls (plain ability) — to see the baseline SetStatus event handler count.
  ['Pressure(par-ctrl)',  'thunderwave', 'Snorlax', false],
  ['Pressure(slp-ctrl)',  'spore',       'Snorlax', false],
  ['Pressure(brn-ctrl)',  'willowisp',   'Snorlax', false],
  ['Pressure(tox-ctrl)',  'toxic',       'Snorlax', false],
];

(async () => {
  for (const fmt of ['gen3customgame', 'gen3ou']) {
    console.log(`\n================ FORMAT ${fmt} ================`);
    for (const [label, mv, species, isSec] of MEMBERS) {
      const ability = label.startsWith('Pressure') ? 'Pressure' : label;
      // For a secondary-frz probe, sweep seeds until the freeze secondary fires.
      let r;
      if (isSec) {
        for (let s = 0; s < 400; s++) {
          r = await probe(fmt, ability, mv, species, [s + 1, 2, 3, 4], true);
          // Fire happened iff a SetStatus/Immunity event for frz occurred with a nonzero relayVar path.
          if (r.events.some((e) => e.eventid === 'Immunity' || e.eventid === 'SetStatus')) break;
        }
      } else {
        r = await probe(fmt, ability, mv, species, [3, 5, 7, 11], false);
      }
      const evStr = r.events.map((e) => `${e.eventid}(handlers=${e.nHandlers},draws=${e.draws},ret=${JSON.stringify(e.ret)})`).join('  ');
      console.log(`  ${label.padEnd(20)} tgtStatus='${r.tgtStatus}'  events: ${evStr || '(none)'}`);
    }
  }
})();
