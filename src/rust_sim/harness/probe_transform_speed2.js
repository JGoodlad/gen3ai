// probe_transform_speed2.js — DIRECT read of the cached `pokemon.speed` at the instant the
// `|-transform|` line is emitted (and at every `speedSort` call), so the ROUND-33 port models
// the right value in the window between the Transform action and the residual's updateSpeed().
// Run: node harness/probe_transform_speed2.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const mon = (species, moves, o = {}) => ({
  species, item: o.item || '', ability: o.ability || 'No Ability', moves,
  evs: { ...EV0, ...(o.evs || {}) }, ivs: o.ivs || IV31, nature: o.nature || 'Serious',
  level: 100, gender: '',
});
const tick = () => new Promise((r) => setTimeout(r, 0));

async function go(teams, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const c of streams.omniscient) void c; })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const b = stream.battle;
  const trace = [];
  const realAdd = b.add.bind(b);
  b.add = (...a) => { if (a[0] === '-transform') trace.push({ at: 'emit -transform', p1: b.sides[0].active[0].speed, p1stored: b.sides[0].active[0].storedStats.spe, p2: b.sides[1].active[0].speed }); return realAdd(...a); };
  const realSort = b.speedSort.bind(b);
  b.speedSort = (list, cmp) => {
    const sp = list.map((x) => (x && x.speed !== undefined ? x.speed : (x && x.effectHolder ? '?' : '?')));
    trace.push({ at: 'speedSort', n: list.length, speeds: sp, p1: b.sides[0].active[0].speed, p2: b.sides[1].active[0].speed });
    return realSort(list, cmp);
  };
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    trace.push({ at: '--- boundary ---' });
    if (b.ended) break;
  }
  return trace;
}

async function main() {
  const g = Dex.mod('gen3');
  console.log(`Chansey base spe=${g.species.get('Chansey').baseStats.spe}  Ditto base spe=${g.species.get('Ditto').baseStats.spe}`);
  const teams = [
    [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
    [mon('Chansey', ['splash'], { ability: 'Natural Cure', ivs: { ...IV31, spe: 27 } })],
  ];
  const t = await go(teams, [5, 4, 3, 2], [['move 1', 'move 1']]);
  for (const e of t) console.log('  ' + JSON.stringify(e));
}
main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
