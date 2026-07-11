// probe_statusimmune_shuffle_size.js — settle the SHUFFLE SIZE + DRAW COUNT directly.
//
// Instruments prng.shuffle AND prng.next to record, for the SetStatus event on an
// onSetStatus-immunity target in gen3ou, the EXACT shuffle range(s) + total next() draws.
// The question: a size-3 handler tie (2 clauses + the ability) — does the Fisher-Yates
// shuffle(list, s, s+3) draw 1 or 2 `random()`s? And how does comparePriority actually
// GROUP the 3 handlers (are all 3 in one tie group, or does the ability sort distinctly)?
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function probe(fmt, ability, mv, species) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":[3,5,7,11]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Blissey', [mv, 'recover'])]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon(species, ['rest', 'bodyslam'], { ability })]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  let inSetStatus = false;
  const shuffleCalls = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };

  const realShuffle = battle.prng.shuffle.bind(battle.prng);
  battle.prng.shuffle = function (items, start, end) {
    if (inSetStatus) {
      const before = nextCount;
      const r = realShuffle(items, start, end);
      shuffleCalls.push({ start, end, size: (end ?? items.length) - (start ?? 0), draws: nextCount - before });
      return r;
    }
    return realShuffle(items, start, end);
  };

  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, ...rest) {
    if (eventid === 'SetStatus') {
      const handlers = battle.findEventHandlers(rest[0], 'SetStatus', rest[1]);
      const detail = handlers.map((h) => {
        const e = h.effect || {};
        return `${e.name || e.id}(order=${h.order},pri=${h.priority},spd=${h.speed},subOrder=${h.subOrder})`;
      });
      const prev = inSetStatus; inSetStatus = true;
      const ret = realRun(eventid, ...rest);
      inSetStatus = prev;
      console.log(`      SetStatus handlers(${handlers.length}): ${detail.join(' | ')}`);
      return ret;
    }
    return realRun(eventid, ...rest);
  };

  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 12; k++) await tick();
  return { shuffleCalls, tgtStatus: (battle.sides[1].active[0].status || '') };
}

(async () => {
  const CASES = [
    ['Limber', 'thunderwave', 'Snorlax'],
    ['Insomnia', 'spore', 'Snorlax'],
    ['Immunity', 'toxic', 'Snorlax'],
    ['Water Veil', 'willowisp', 'Snorlax'],
    ['Pressure(ctrl)', 'thunderwave', 'Snorlax'],
  ];
  for (const fmt of ['gen3ou', 'gen3customgame']) {
    console.log(`\n=== ${fmt} ===`);
    for (const [ability, mv, species] of CASES) {
      const ab = ability.startsWith('Pressure') ? 'Pressure' : ability;
      console.log(`  ${ability} [${mv}]:`);
      const r = await probe(fmt, ab, mv, species);
      console.log(`      shuffles: ${JSON.stringify(r.shuffleCalls)}  tgtStatus='${r.tgtStatus}'`);
    }
  }
})();
