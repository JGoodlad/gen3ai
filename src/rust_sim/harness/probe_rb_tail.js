// probe_rb_tail.js — SIM ORACLE for the RANDBATS byte-fuzz emission tail (round: rb).
//
// Drives the OMNISCIENT in-process BattleStream (no server, resolved Dex.mod('gen3')) over
// CONSTRUCTED gen3customgame boards and prints the RAW `|...|` protocol lines per decision, so the
// exact byte FORM + ORDER of each cluster is settled by the sim, never guessed.
//
// Clusters:
//   C1  Leech Seed x Protect         — TryHit (Protect) vs naturalImmunity (Grass) report order
//   C2  White Herb x on-hit proc     — onAnyAfterMove timing vs the DamagingHit-phase procs
//   C3  Fire Punch frz-thaw x Static — the two onDamagingHit handlers' order (status vs ability)
//   S1  Wonder Guard x Solar Beam    — a sun-skipped charge move into a WG holder
//   S2  Trace->Flash Fire x WoW      — a WoW into a TRACED Flash Fire
//   S3  Liquid Ooze x Leech Seed     — the DOUBLE-FAINT emit order
//
// Run:  node src/rust_sim/harness/probe_rb_tail.js [filter]
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan, prep) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 14; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  if (prep) { prep(b); await tick(); }
  console.log(`\n================ ${label} (raw seed ${rawSeed.join(',')}) ================`);
  let mark = log.length;
  let i = 0;
  for (const entry of plan) {
    if (entry.pre) { entry.pre(b); await tick(); }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 22; k++) await tick();
    console.log(`--- dec ${i} ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    for (const l of log.slice(mark)) {
      if (l.startsWith('|t:|') || l.startsWith('|debug|')) continue;
      console.log(`    ${l}`);
    }
    mark = log.length;
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

const SEED = [11, 29, 37, 53];
const filter = process.argv[2] || '';
const want = (t) => !filter || t.toLowerCase().includes(filter.toLowerCase());

async function main() {
  // ---------------- C1: Leech Seed x Protect ----------------
  if (want('c1')) {
    const cacturne = "Cacturne||leftovers|Sand Veil|leechseed,splash|Hardy|252,,,,,|N||||";
    const jumpluff = "Jumpluff||leftovers|Chlorophyll|protect,splash|Hardy|252,,,,,|N||||";  // Grass/Flying
    const snorlax = "Snorlax||leftovers|Own Tempo|protect,splash|Hardy|252,,,,,|N||||";      // non-Grass
    await run('C1a Leech Seed into a PROTECTING GRASS foe', cacturne, jumpluff, SEED, [
      { p1: 'move 1', p2: 'move 1' },   // p2 Protect, p1 Leech Seed
    ]);
    await run('C1b Leech Seed into a NON-protecting GRASS foe (control)', cacturne, jumpluff, SEED, [
      { p1: 'move 1', p2: 'move 2' },   // p2 Splash, p1 Leech Seed
    ]);
    await run('C1c Leech Seed into a PROTECTING NON-GRASS foe (control)', cacturne, snorlax, SEED, [
      { p1: 'move 1', p2: 'move 1' },
    ]);
  }

  // ---------------- C2: White Herb x on-hit proc ----------------
  if (want('c2')) {
    // A White-Herb Superpower user into a Poison Point / Rough Skin holder.
    const whSuper = "Machamp||whiteherb|Guts|superpower,splash|Hardy|252,,,,,|M||||";
    const ppFoe = "Nidoqueen||leftovers|Poison Point|splash|Hardy|252,,252,,,|F||||";
    const rsFoe = "Sharpedo||leftovers|Rough Skin|splash|Hardy|252,,252,,,|M||||";
    // Sweep several seeds so the 1/3 contact-proc roll lands.
    for (const s of [[11, 29, 37, 53], [3, 5, 7, 11], [21, 44, 88, 99], [1, 2, 3, 4]]) {
      await run(`C2a Superpower(White Herb) into POISON POINT seed=${s.join(',')}`, whSuper, ppFoe, s, [
        { p1: 'move 1', p2: 'move 1' },
      ]);
    }
    await run('C2b Superpower(White Herb) into ROUGH SKIN', whSuper, rsFoe, SEED, [
      { p1: 'move 1', p2: 'move 1' },
    ]);
  }

  // ---------------- C3: Fire Punch frz-thaw x Static ----------------
  if (want('c3')) {
    const jirachi = "Jirachi||leftovers|Serene Grace|firepunch,splash|Hardy|252,252,,,,|N||||";
    const pika = "Pikachu||leftovers|Static|splash|Hardy|252,,252,,,|M||||";
    for (const s of [[11, 29, 37, 53], [3, 5, 7, 11], [21, 44, 88, 99], [1, 2, 3, 4], [77, 13, 5, 2]]) {
      await run(`C3 Fire Punch into a FROZEN STATIC foe seed=${s.join(',')}`, jirachi, pika, s, [
        { p1: 'move 1', p2: 'move 1' },
      ], (b) => {
        const t = b.sides[1].active[0];
        t.setStatus('frz');
        b.log.length = b.log.length; // keep
      });
    }
  }

  // ---------------- S1: Wonder Guard x Solar Beam ----------------
  if (want('s1')) {
    const tangela = "Tangela||leftovers|Chlorophyll|sunnyday,solarbeam,splash|Hardy|252,,,252,,|N||||";
    const shedinja = "Shedinja||leftovers|Wonder Guard|splash|Hardy|252,,,,,|N||||";
    await run('S1a Solar Beam (SUN, charge skipped) into WONDER GUARD', shedinja, tangela, SEED, [
      { p1: 'move 1', p2: 'move 1' },   // p2 Sunny Day
      { p1: 'move 1', p2: 'move 2' },   // p2 Solar Beam -> fires immediately in sun
    ]);
    await run('S1b Solar Beam (NO sun, 2-turn) into WONDER GUARD', shedinja, tangela, SEED, [
      { p1: 'move 1', p2: 'move 2' },
      { p1: 'move 1', p2: 'move 2' },
    ]);
    // Control: a SUPER-EFFECTIVE charge move (none in gen3 vs Bug/Ghost by solarbeam) — use a
    // NEUTRAL non-charge move to show the plain WG immune form for contrast.
    const control = "Tangela||leftovers|Chlorophyll|absorb,splash|Hardy|252,,,252,,|N||||";
    await run('S1c plain (non-charge) move into WONDER GUARD (control)', shedinja, control, SEED, [
      { p1: 'move 1', p2: 'move 1' },
    ]);
  }

  // ---------------- S2: Trace -> Flash Fire x Will-O-Wisp ----------------
  if (want('s2')) {
    const houndoom = "Houndoom||leftovers|Flash Fire|willowisp,splash|Hardy|252,,,,,|M||||";
    const gardevoir = "Gardevoir||leftovers|Trace|splash|Hardy|252,,252,,,|F||||";
    await run('S2 Will-O-Wisp into a TRACED Flash Fire', houndoom, gardevoir, SEED, [
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]);
  }

  // ---------------- S3: Liquid Ooze x Leech Seed DOUBLE FAINT ----------------
  if (want('s3')) {
    const jumpluff2 = "Jumpluff||leftovers|Chlorophyll|leechseed,splash|Hardy|252,,,,,|N||||";
    const swalot = "Swalot||leftovers|Liquid Ooze|splash|Hardy|252,,252,,,|M||||";
    const lowBoth = (b) => {
      // Swalot (seeded) takes floor(404/8)=50 -> set it to 40 so the drain KOs it;
      // Jumpluff (seeder) then takes the REVERSED heal (40 clamped) -> set it to 30 so it also dies.
      b.sides[1].active[0].hp = 20;   // Swalot (seeded): +25 lefto, -50 leech -> 0 fnt (45 dealt)
      b.sides[0].active[0].hp = 20;   // Jumpluff (seeder): +22 lefto, -45 ooze -> 0 fnt
    };
    await run('S3 Liquid Ooze x Leech Seed DOUBLE FAINT', jumpluff2, swalot, SEED, [
      { p1: 'move 1', p2: 'move 1' },     // seed it
      { p1: 'move 2', p2: 'move 1', pre: lowBoth },  // residual double-KOs
    ], null);
    // Reverse order check: the seeder survives, only the seeded dies.
    await run('S3b Liquid Ooze x Leech Seed — only the SEEDED faints (control)', jumpluff2, swalot, SEED, [
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1', pre: (b) => { b.sides[1].active[0].hp = 40; } },
    ], null);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
