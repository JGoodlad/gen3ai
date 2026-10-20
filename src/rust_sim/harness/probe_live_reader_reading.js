// PROBE: the live reader's reading fixes (`gen3_live_reader_reading_v1`) — scripted gen3customgame
// battles on the PINNED Showdown, each PLAYER stream (what the live reader is fed: the battle lines
// plus that side's `|request|`, `|t:|` dropped) written one protocol line per text line to
// `tests/vectors/live_reader_reading/<scenario>_<p1|p2>.txt`. `present::live_reader_reading_tests`
// reads them. Regenerate (and review the diff) with:
//
//   node src/rust_sim/harness/probe_live_reader_reading.js --write
//
// Scenario  seed       what the stream holds
//   spite     [1,2,3,4] Spite cuts Snorlax's Body Slam (`-activate|…|move: Spite|bodyslam|4`), then
//                       Lock-On (turn 4) and Mind Reader (turns 6 and 9) from Electrode
//   mimic     [1,2,3,4] Snorlax (HP Dark by its IVs) Mimics Jolteon's Hidden Power
//   yawn      [1,2,3,4] Slowbro Yawns Snorlax, which Rests the same turn and wakes on its Chesto Berry
//   sleeptalk [4,2,3,4] Snorlax's Sleep Talk calls Body Slam into a Pressure Zapdos
//   focusband [9,2,3,4] Abra survives a crit Cross Chop on its Focus Band
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const fs = require('fs');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Teams } = require(path.join(PS, 'dist/sim/teams.js'));
const tick = () => new Promise((r) => setImmediate(r));
function mon(sp, mv, o = {}) {
  return { species: sp, name: sp, item: o.item || '', ability: o.ability || '', moves: mv, nature: o.nature || 'Hardy',
    evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: o.spe === undefined ? 85 : o.spe }, gender: o.gender || 'M',
    ivs: o.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: o.level || 100 };
}
const SC = {
  spite: { p1: [mon('Gengar', ['spite', 'lockon', 'mindreader', 'shadowball'], { ability: 'levitate', spe: 0 }), mon('Electrode', ['lockon', 'zapcannon', 'mindreader', 'thunderbolt'], { ability: 'static' })],
           p2: [mon('Snorlax', ['bodyslam', 'curse', 'rest', 'sleeptalk'], { ability: 'immunity', spe: 0, level: 100 }), mon('Blissey', ['softboiled', 'seismictoss', 'toxic', 'aromatherapy'], { ability: 'naturalcure' })],
           script: [['move 4', 'move 1'], ['move 1', 'move 1'], ['switch 2', 'move 2'], ['move 1', 'move 2'], ['move 2', 'move 2'], ['move 3', 'move 2'], ['move 4', 'move 2'], ['move 4', 'move 2'], ['move 3', 'move 2']] },
  mimic: { p1: [mon('Snorlax', ['mimic', 'curse', 'rest', 'bodyslam'], { ability: 'immunity', spe: 0 })],
           p2: [mon('Jolteon', ['hiddenpower', 'substitute', 'batonpass', 'agility'], { ability: 'voltabsorb', ivs: { hp: 30, atk: 31, def: 31, spa: 30, spd: 31, spe: 30 } })],
           script: [['move 2', 'move 1'], ['move 1', 'move 1'], ['move 4', 'move 4'], ['move 4', 'move 4']] },
  yawn: { p1: [mon('Snorlax', ['rest', 'curse', 'bodyslam', 'sleeptalk'], { ability: 'immunity', item: 'chestoberry', spe: 0 })],
          p2: [mon('Slowbro', ['yawn', 'surf', 'calmmind', 'thunderwave'], { ability: 'owntempo', })],
          script: [['move 2', 'move 2'], ['move 1', 'move 1'], ['move 2', 'move 3'], ['move 2', 'move 3']] },
  sleeptalk: { p1: [mon('Snorlax', ['rest', 'sleeptalk', 'bodyslam', 'return'], { ability: 'immunity', spe: 0 }), mon('Skarmory', ['spikes', 'whirlwind', 'drillpeck', 'toxic'], { ability: 'keeneye' })],
               p2: [mon('Zapdos', ['thunderbolt', 'drillpeck', 'agility', 'substitute'], { ability: 'pressure', })],
               script: [['move 3', 'move 1'], ['move 1', 'move 3'], ['move 2', 'move 3'], ['switch 2', 'move 3'], ['move 1', 'move 3']] },
  focusband: { p1: [mon('Machamp', ['crosschop', 'bulkup', 'rockslide', 'earthquake'], { ability: 'guts', item: 'choiceband', level: 100 })],
               p2: [mon('Abra', ['teleport', 'calmmind', 'psychic', 'substitute'], { ability: 'synchronize', item: 'focusband', spe: 0, level: 5 }), mon('Blissey', ['softboiled', 'seismictoss', 'toxic', 'aromatherapy'], { ability: 'naturalcure' })],
               script: [['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']] },
};
async function run(name, seed) {
  const sc = SC[name];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const out = { p1: [], p2: [] };
  for (const s of ['p1', 'p2']) {
    (async () => { for await (const ch of streams[s]) out[s].push(ch.split('\n').filter((l) => !l.startsWith('|t:|'))); })();
  }
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'me', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'foe', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 20; i++) await tick();
  for (const [c1, c2] of sc.script) {
    if (stream.battle.ended) break;
    const b = stream.battle;
    if (b.sides[0].activeRequest && !b.sides[0].activeRequest.wait) streams.omniscient.write(`>p1 ${c1}`);
    for (let i = 0; i < 5; i++) await tick();
    if (b.sides[1].activeRequest && !b.sides[1].activeRequest.wait) streams.omniscient.write(`>p2 ${c2}`);
    for (let i = 0; i < 20; i++) await tick();
  }
  return { p1: Teams.pack(sc.p1), p2: Teams.pack(sc.p2), out };
}
const SEEDS = { spite: [1, 2, 3, 4], mimic: [1, 2, 3, 4], yawn: [1, 2, 3, 4], sleeptalk: [4, 2, 3, 4], focusband: [9, 2, 3, 4] };
(async () => {
  const write = process.argv.includes('--write');
  const dir = path.resolve(__dirname, '../tests/vectors/live_reader_reading');
  for (const [name, seed] of Object.entries(SEEDS)) {
    const r = await run(name, seed);
    console.log(`${name} seed ${JSON.stringify(seed)}\n  p1 team ${r.p1}\n  p2 team ${r.p2}`);
    for (const s of ['p1', 'p2']) {
      const text = r.out[s].flat().join('\n') + '\n';
      const f = path.join(dir, `${name}_${s}.txt`);
      if (write) fs.writeFileSync(f, text);
      else if (!fs.existsSync(f) || fs.readFileSync(f, 'utf8') !== text) console.log(`  ${s}: DIFFERS from ${f}`);
      else console.log(`  ${s}: matches ${f}`);
    }
  }
  process.exit(0);
})();
