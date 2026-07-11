// probe_f1_f2_f3_lines.js — nail the EXACT omniscient line forms for protocol
// review findings F1-F3 (emission-layer). The resolved Dex.mod('gen3') sim is the
// ONLY oracle. Dumps the raw |...| stream for each scenario across a seed sweep so
// both the MISS and HIT branches (F2/F3) and the sub-block branch (F1) realize.
//
//   F1 — a Leech Seed into a SUBSTITUTE'd foe: the sub blocks the volatile. Expect
//        `|move|<user>|Leech Seed||[still]` + `|-fail|<user>` (mirror already-seeded).
//   F2 — a MISSED Fire move into Flash Fire (and Water into Water Absorb): gen3 rolls
//        accuracy BEFORE TryHit, so a MISS shows `|move|...|[miss]` + `|-miss|...`,
//        NOT `-immune`.
//   F3 — a LANDED Water/Volt Absorb absorb: `|-immune|<t>|[from] ability: Water Absorb`.
//
// Run:  node src/rust_sim/harness/probe_f1_f2_f3_lines.js
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

async function run(label, p1team, p2team, plan, seed) {
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

  let d = 0;
  let safety = 0;
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
  return lines;
}

function grep(lines, res) {
  return lines.filter((l) => res.some((r) => l.includes(r)));
}

async function main() {
  // ── F1 — Leech Seed into a substituted foe ──
  // p2 Suicune subs turn 1; p1 Venusaur Leech Seeds the sub turn 2.
  const f1p1 = [mon('Venusaur', ['leechseed', 'sludgebomb'], { ability: 'Overgrow', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  const f1p2 = [mon('Suicune', ['substitute', 'surf'], { item: 'Leftovers', ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  const f1 = await run('F1', f1p1, f1p2, [
    { p1: 'move 2', p2: 'move 1' },  // Suicune subs (Venusaur sludgebomb chips)
    { p1: 'move 1', p2: 'move 2' },  // Venusaur Leech Seed into the sub → BLOCKED
    { p1: 'move 1', p2: 'move 2' },  // Leech Seed again into sub
  ], [1, 2, 3, 4]);
  console.log('=== F1: Leech Seed into a Substitute (seed 1,2,3,4) ===');
  for (const l of grep(f1, ['Leech Seed', '-fail', 'Substitute', '-activate', '-start'])) console.log('  ' + l);

  // ── F2/F3 — Fire into Flash Fire + Water into Water Absorb, seed-swept ──
  // p1 Moltres Fire Blast (85 acc) into p2 Houndoom (Flash Fire).
  // p1 also Suicune Hydro Pump (80 acc) into p2 Politoed (Water Absorb).
  const f2p1 = [mon('Moltres', ['fireblast', 'hiddenpowergrass'], { nature: 'Modest', evs: { spa: 252, spe: 252 } }),
               mon('Suicune', ['hydropump', 'icebeam'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })];
  const f2p2 = [mon('Houndoom', ['crunch', 'flamethrower'], { ability: 'Flash Fire', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
               mon('Politoed', ['icebeam', 'surf'], { item: 'Leftovers', ability: 'Water Absorb', nature: 'Bold', evs: { hp: 252, def: 252 } })];
  // Sweep seeds so both a MISS and a HIT of Fire Blast into FF, and Hydro Pump into WA, realize.
  const seeds = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [2, 4, 8, 16], [3, 9, 27, 81]];
  console.log('\n=== F2/F3: Fire Blast into Flash Fire (85 acc) across seeds ===');
  for (const s of seeds) {
    const ls = await run('F2-FF', f2p1, f2p2, [{ p1: 'move 1', p2: 'move 2' }], s);
    const rel = grep(ls, ['Fire Blast', '-immune', '-miss', 'Flash Fire', '[miss]', '[still]']);
    console.log(`  seed ${s.join(',')}: ` + JSON.stringify(rel));
  }
  console.log('\n=== F2/F3: Hydro Pump into Water Absorb (80 acc) across seeds ===');
  for (const s of seeds) {
    const ls = await run('F2-WA', f2p1, f2p2, [{ p1: 'switch 2', p2: 'switch 2' }, { p1: 'move 1', p2: 'move 1' }], s);
    const rel = grep(ls, ['Hydro Pump', '-immune', '-miss', 'Water Absorb', '[miss]']);
    console.log(`  seed ${s.join(',')}: ` + JSON.stringify(rel));
  }
}
main();
