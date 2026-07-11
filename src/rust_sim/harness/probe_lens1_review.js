// probe_lens1_review.js — INDEPENDENT adversarial Lens-1 capture for protocol
// findings F1/F2/F3, using DIFFERENT species + seeds than the builder's probes.
// The resolved Dex.mod('gen3') sim is the ONLY oracle. For each scenario+seed it
// prints the raw |...| stream AND the POST-SWITCHIN `getSeed()` (== the protocol
// golden's `initSeed` — the seed the Rust port must be fed, since gen3 switch-ins
// ADVANCE the PRNG). The Rust byte-compare uses those printed INIT seeds.
//
//   F1 — Leech Seed into a SUBSTITUTE'd foe (Meganium → Milotic-sub):
//        `|move|<user>|Leech Seed||[still]` + `|-fail|<user>`.
//   F2 — Fire into Flash Fire (Charizard Fire Blast → Ninetales): HIT arms FF (no
//        -immune); a MISS shows `[miss]`+`-miss`. Water into Water Absorb
//        (Vaporeon Hydro Pump → Lapras-WA): landed `-immune|[from] ability: Water Absorb`.
//   F3 — LANDED Volt Absorb (Zapdos Thunder → Lanturn-VoltAbsorb):
//        `|-immune|<t>|[from] ability: Volt Absorb`.
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Returns { lines, initSeed } — initSeed = the POST-SWITCHIN getSeed() the port needs.
async function run(p1team, p2team, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) {
        if (l && !l.startsWith('|t:|') && !l.startsWith('|split') && l !== '|') lines.push(l);
      }
    }
  })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const initSeed = stream.battle.prng.getSeed(); // post-switchin (draw-free) — the Rust seed

  let d = 0, safety = 0;
  while (!stream.battle.ended && safety < 60 && d < plan.length) {
    safety++;
    const rs = stream.battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const step = plan[d];
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();
    d++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { lines, initSeed: String(initSeed) };
}
function grep(lines, res) { return lines.filter((l) => res.some((r) => l.includes(r))); }

async function main() {
  // ── F1 — Meganium Leech Seed into a Milotic Substitute ──
  const f1p1 = [mon('Meganium', ['leechseed', 'bodyslam'], { ability: 'Overgrow', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  const f1p2 = [mon('Milotic', ['substitute', 'surf'], { item: 'Leftovers', ability: 'Marvel Scale', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  console.log('PACK_F1_P1 ' + Teams.pack(f1p1));
  console.log('PACK_F1_P2 ' + Teams.pack(f1p2));
  for (const s of [[7, 11, 13, 17], [23, 29, 31, 37]]) {
    const { lines, initSeed } = await run(f1p1, f1p2, [
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 2' },
    ], s);
    console.log(`\n=== F1 start ${s.join(',')} INIT ${initSeed} (Leech Seed into Sub) ===`);
    for (const l of grep(lines, ['Leech Seed', '-fail', 'Substitute', '-activate', '-start', '[still]', '[miss]', '-miss', '-immune'])) console.log('  ' + l);
  }

  // ── F2 — Charizard Fire Blast into Ninetales Flash Fire (85 acc); p2 uses a MODELED move ──
  const f2p1 = [mon('Charizard', ['fireblast', 'aerialace'], { nature: 'Modest', evs: { spa: 252, spe: 252 } }),
               mon('Vaporeon', ['hydropump', 'icebeam'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })];
  const f2p2 = [mon('Ninetales', ['flamethrower', 'quickattack'], { ability: 'Flash Fire', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
               mon('Lapras', ['icebeam', 'bodyslam'], { item: 'Leftovers', ability: 'Water Absorb', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  console.log('\nPACK_F2_P1 ' + Teams.pack(f2p1));
  console.log('PACK_F2_P2 ' + Teams.pack(f2p2));
  const seeds = [[7,11,13,17],[23,29,31,37],[41,43,47,53],[59,61,67,71],[2,4,8,16],[3,9,27,81]];
  console.log('\n=== F2: Charizard Fire Blast into Ninetales Flash Fire (85 acc) ===');
  for (const s of seeds) {
    const { lines, initSeed } = await run(f2p1, f2p2, [{ p1: 'move 1', p2: 'move 2' }], s);
    console.log(`  INIT ${initSeed}: ` + JSON.stringify(grep(lines, ['Fire Blast', '-immune', '-miss', 'Flash Fire', '[miss]', '[still]', '-start'])));
  }
  console.log('\n=== F2: Vaporeon Hydro Pump into Lapras Water Absorb (80 acc) ===');
  for (const s of seeds) {
    const { lines, initSeed } = await run(f2p1, f2p2, [{ p1: 'switch 2', p2: 'switch 2' }, { p1: 'move 1', p2: 'move 2' }], s);
    console.log(`  INIT ${initSeed}: ` + JSON.stringify(grep(lines, ['Hydro Pump', '-immune', '-miss', 'Water Absorb', '[miss]'])));
  }

  // ── F3 — Zapdos Thunder into Lanturn Volt Absorb (LANDED) ──
  const f3p1 = [mon('Zapdos', ['thunder', 'drillpeck'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })];
  const f3p2 = [mon('Lanturn', ['surf', 'icebeam'], { item: 'Leftovers', ability: 'Volt Absorb', nature: 'Calm', evs: { hp: 252, spd: 252 } })];
  console.log('\nPACK_F3_P1 ' + Teams.pack(f3p1));
  console.log('PACK_F3_P2 ' + Teams.pack(f3p2));
  console.log('\n=== F3: Zapdos Thunder into Lanturn Volt Absorb ===');
  for (const s of [[23,29,31,37],[41,43,47,53],[7,11,13,17]]) {
    const { lines, initSeed } = await run(f3p1, f3p2, [{ p1: 'move 1', p2: 'move 1' }], s);
    console.log(`  INIT ${initSeed}: ` + JSON.stringify(grep(lines, ['Thunder', '-immune', '-miss', 'Volt Absorb', '[miss]', '-heal'])));
  }
}
main();
